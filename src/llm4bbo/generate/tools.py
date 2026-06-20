from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizerBase
from transformers.pipelines.text_generation import ChatType
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    parse_response
)

from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob

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
        self.chat_template = get_training_chat_template(self.tokenizer)

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

        pending = [
            (request, completion, completion.token_ids)
            for request in requests
            for completion in request.outputs
        ]

        for _ in range(self.max_tool_calling_iterations):
            new_prompts = []
            request_completions = []

            for request, completion, new_completion_ids in pending:
                parsed = parse_response(self.tokenizer, new_completion_ids)
                tool_calls = parsed.get("tool_calls")

                if not tool_calls:
                    continue

                tool_messages = self._execute_tool_calls(tool_calls)
                suffix_ids = self._get_tool_suffix_ids(tool_messages)

                completion.text += self.tokenizer.decode(suffix_ids)
                completion.token_ids += suffix_ids

                if completion.logprobs is not None:
                    completion.logprobs += [
                        {i: Logprob(0.0, decoded_token=self.tokenizer.decode(i))}
                        for i in suffix_ids
                    ]

                new_prompt_ids = request.prompt_token_ids + completion.token_ids
                new_prompts.append({"prompt_token_ids": new_prompt_ids})
                request_completions.append((request, completion))

            if not new_prompts:
                break

            params = sampling_params.clone()
            params.n = 1
            new_requests = super()._colocate_call(new_prompts, params, **kwargs)

            pending = []

            for new_request, (request, completion) in zip(
                new_requests, request_completions, strict=True
            ):
                new_completion = new_request.outputs[0]

                completion.text += new_completion.text
                completion.token_ids += new_completion.token_ids
                completion.finish_reason = new_completion.finish_reason
                completion.stop_reason = new_completion.stop_reason

                if completion.cumulative_logprob is not None:
                    completion.cumulative_logprob += new_completion.cumulative_logprob

                if completion.logprobs is not None:
                    completion.logprobs += new_completion.logprobs

                pending.append((request, completion, new_completion.token_ids))

        return requests

    def _server_call(
        self,
        prompts: list[list[int]],
        n: int = 1,
        **kwargs: Any
    ) -> ServerOutput:
        outputs = super()._server_call(prompts, n, **kwargs)
        pending = [(index, ids) for index, ids in enumerate(outputs["completion_ids"])]

        for _ in range(self.max_tool_calling_iterations):
            new_prompts = []
            indices = []

            for index, new_completion_ids in pending:
                parsed = parse_response(self.tokenizer, new_completion_ids)
                tool_calls = parsed.get("tool_calls")

                if not tool_calls:
                    continue

                tool_messages = self._execute_tool_calls(tool_calls)
                suffix_ids = self._get_tool_suffix_ids(tool_messages)

                outputs["completion_ids"][index] += suffix_ids
                outputs["logprobs"][index] += [[0.0] for _ in range(len(suffix_ids))]
                outputs["logprob_token_ids"][index] += [[i] for i in suffix_ids]

                new_prompt_ids = (
                    outputs["prompt_ids"][index // n] + outputs["completion_ids"][index]
                )
                new_prompts.append(new_prompt_ids)
                indices.append(index)

            if not new_prompts:
                break

            new_output = super()._server_call(new_prompts, **kwargs)
            pending = []

            for new_index, index in enumerate(indices):
                outputs["completion_ids"][index] += (
                    new_output["completion_ids"][new_index]
                )
                outputs["logprobs"][index] += new_output["logprobs"][new_index]
                outputs["logprob_token_ids"][index] += (
                    new_output["logprob_token_ids"][new_index]
                )
                pending.append((index, new_output["completion_ids"][new_index]))

        return outputs

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
            dummy_messages, chat_template=self.chat_template, return_dict=False
        )
        full_ids = self.tokenizer.apply_chat_template(
            dummy_messages + tool_messages,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            return_dict=False
        )

        eos_positions = [i for i, p in enumerate(prefix_ids) if p == self.eos_token_id]

        if eos_positions:
            prefix_ids = prefix_ids[: eos_positions[-1] + 1]

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Unexpected tokenization: "
                "the EOS-trimmed prefix IDs are not a prefix of the full IDs."
            )

        return full_ids[len(prefix_ids) :]
