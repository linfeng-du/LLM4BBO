import copy
from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizerBase
from transformers.pipelines.text_generation import ChatType
from trl.chat_template_utils import add_response_schema, parse_response

from vllm import LLM, SamplingParams

from .budgets import GenerateWithBudgets
from .typings import ColocateOutput, ServerOutput


class GenerateWithTools(GenerateWithBudgets):
    def __init__(
        self,
        generate_func: Callable[..., ColocateOutput | ServerOutput],
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Callable],
        thinking_budget: int,
        answer_budget: int,
        max_tool_calling_iterations: int
    ) -> None:
        super().__init__(generate_func, tokenizer, thinking_budget, answer_budget)
        self.tools = tools
        self.tool_dict = {tool.__name__: tool for tool in tools}
        self.max_tool_calling_iterations = max_tool_calling_iterations

        if self.tokenizer.response_schema is None:
            self.tokenizer = add_response_schema(self.tokenizer)

    def __call__(self, prompts: list[ChatType], *args: Any, **kwargs: Any) -> (
        ColocateOutput | ServerOutput
    ):
        prompt_ids = [
            self.tokenizer.apply_chat_template(
                p, self.tools, add_generation_prompt=True, return_dict=False
            )
            for p in prompts
        ]

        if isinstance(self.generate_func.__self__, LLM):
            prompt_ids = [{"prompt_token_ids": ids} for ids in prompt_ids]

        return super().__call__(prompt_ids, *args, **kwargs)

    def _colocate_call(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> ColocateOutput:
        requests = super()._colocate_call(prompts, sampling_params, **kwargs)

        for _ in range(self.max_tool_calling_iterations):
            stage_prompts = []
            completions = []

            for request in requests:
                for completion in request.outputs:
                    parsed = parse_response(self.tokenizer, completion.token_ids)
                    tool_calls = parsed.get("tool_calls")

                    if not tool_calls:
                        continue

                    tool_messages = self._execute_tool_calls(tool_calls)
                    suffix_ids = self._get_tool_suffix_ids(tool_messages)

                    completion.text += self.tokenizer.decode(suffix_ids)
                    completion.token_ids += suffix_ids

                    prompt_ids = request.prompt_token_ids + completion.token_ids
                    stage_prompts.append({"prompt_token_ids": prompt_ids})
                    completions.append(completion)

            if not stage_prompts:
                break

            stage_params = sampling_params.clone()
            stage_params.n = 1
            stage_requests = super()._colocate_call(
                stage_prompts, stage_params, **kwargs
            )

            for stage_request, completion in zip(
                stage_requests, completions, strict=True
            ):
                stage_completion = stage_request.outputs[0]
                completion.text += stage_completion.text
                completion.token_ids += stage_completion.token_ids
                completion.finish_reason = stage_completion.finish_reason
                completion.stop_reason = stage_completion.stop_reason

        return requests

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> (
        list[dict[str, Any]]
    ):
        # https://github.com/huggingface/trl/blob/v1.5.1/trl/trainer/grpo_trainer.py#L1514
        tool_call_results = []

        for tool_call in tool_calls:
            if tool_call["type"] == "function":
                function = tool_call["function"]
                name = function["name"]

                try:
                    if name not in self.tool_dict:
                        raise ValueError(f"Tool {name} not found.")

                    result = self.tool_dict[name](**function["arguments"])
                    tool_call_results.append((name, result))

                except Exception as e:
                    result = {"error": str(e)}
                    tool_call_results.append((name, result))

            else:
                name = tool_call.get("name", "unknown")
                result = {"error": f"Unsupported tool call type: {tool_call['type']}"}
                tool_call_results.append((name, result))

        return [
            {"role": "tool", "name": name, "content": str(result)}
            for name, result in tool_call_results
        ]

    # https://github.com/huggingface/trl/blob/v1.5.1/trl/trainer/grpo_trainer.py#L1433
    def _get_tool_suffix_ids(self, tool_messages: list[dict[str, Any]]) -> list[int]:
        dummy_tool_calls = [
            {
                "type": "function",
                "function": {"name": tool_messages[0]["name"], "arguments": {}}
            }
        ]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {"role": "assistant", "content": "", "tool_calls": dummy_tool_calls}
        ]

        prefix_ids = self.tokenizer.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False
        )
        full_ids = self.tokenizer.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False
        )

        num_eos = sum(1 for i in prefix_ids if i == self.eos_token_id)
        eos_positions = [i for i, fi in enumerate(full_ids) if fi == self.eos_token_id]
        return full_ids[eos_positions[num_eos - 1] + 1 :]

    def _append_suffix_to_server_completion(
        self,
        output: ServerOutput,
        completion_index: int,
        suffix_ids: list[int],
    ) -> None:
        output["completion_ids"][completion_index] += suffix_ids

    def _server_call(
        self,
        prompts: list[list[int]],
        **kwargs: Any,
    ) -> ServerOutput:
        kwargs = copy.deepcopy(kwargs)
        kwargs["generation_kwargs"] = kwargs.get("generation_kwargs") or {}

        output = super()._server_call(prompts, **kwargs)
        n = kwargs.get("n", 1)

        for _ in range(self.max_tool_calling_iterations):
            idxs_with_tool: list[int] = []

            for completion_index, completion_ids in enumerate(output["completion_ids"]):
                parsed = parse_response(self.tokenizer, completion_ids)
                if parsed.get("tool_calls"):
                    idxs_with_tool.append(completion_index)

            if not idxs_with_tool:
                break

            stage_prompts: list[list[int]] = []

            for completion_index in idxs_with_tool:
                parsed = parse_response(
                    self.tokenizer, output["completion_ids"][completion_index]
                )
                tool_messages = self._execute_tool_calls(parsed["tool_calls"])
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                self._append_suffix_to_server_completion(
                    output, completion_index, suffix_ids
                )
                stage_prompts.append(
                    output["prompt_ids"][completion_index // n]
                    + output["completion_ids"][completion_index]
                )

            stage_kwargs = copy.deepcopy(kwargs)
            stage_kwargs["n"] = 1
            stage_output = self.generate_func(stage_prompts, **stage_kwargs)

            for stage_index, completion_index in enumerate(idxs_with_tool):
                output["completion_ids"][completion_index] += stage_output[
                    "completion_ids"
                ][stage_index]

        return output
