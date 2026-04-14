import copy
from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizerBase
from trl.generation.vllm_client import VLLMClient

from vllm import LLM, RequestOutput, SamplingParams
from vllm.logprobs import Logprob


STOP_THINKING_PROMPT = """

Considering the limited time by the user, \
I have to give the solution based on the thinking directly now.
</think>

"""


ServerGenerateOutput = dict[str, list[list[int]] | list[list[float]]]
ColocateGenerateOutput = list[RequestOutput]


class ThinkingBudgetVLLMGenerate:
    def __init__(
        self,
        generate: Callable[..., ServerGenerateOutput | ColocateGenerateOutput],
        tokenizer: PreTrainedTokenizerBase,
        thinking_budget: int
    ) -> None:
        self.generate = generate
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget

        self.eos_token_id = self.tokenizer.eos_token_id
        assert self.eos_token_id is not None

        self.eoth_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        assert self.eoth_token_id != self.tokenizer.unk_token_id

        self.stop_thinking_ids = self.tokenizer.encode(
            STOP_THINKING_PROMPT, add_special_tokens=False
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> ServerGenerateOutput | ColocateGenerateOutput:
        if isinstance(self.generate.__self__, VLLMClient):
            return self._server_call(*args, **kwargs)
        elif isinstance(self.generate.__self__, LLM):
            return self._colocate_call(*args, **kwargs)
        else:
            raise TypeError(
                f"Invalid type for `self.generate.__self__`: "
                f"{type(self.generate.__self__)}"
            )

    def _server_call(
        self,
        prompts: list[list[int]],
        **kwargs: Any
    ) -> ServerGenerateOutput:
        assert "stop" not in kwargs["generation_kwargs"]
        assert "stop_token_ids" not in kwargs["generation_kwargs"]
        max_tokens = kwargs["max_tokens"]

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_kwargs = copy.deepcopy(kwargs)
        stage_1_kwargs["max_tokens"] = (
            self.thinking_budget - len(self.stop_thinking_ids)
        )
        stage_1_kwargs["generation_kwargs"]["stop"] = ["</think>\n\n"]

        stage_1_output = self.generate(prompts, **stage_1_kwargs)

        # Stage 2: Generate the final answer
        stage_2_prompts = []

        for completion_index, (completion_ids, logprobs) in enumerate(
            zip(
                stage_1_output["completion_ids"],
                stage_1_output["logprobs"],
                strict=True
            )
        ):
            assert self.eos_token_id not in completion_ids

            if self.eoth_token_id not in completion_ids:
                # Stop the thinking by inserting `self.stop_thinking_ids`
                completion_ids += self.stop_thinking_ids
                logprobs += [[0.0]] * len(self.stop_thinking_ids)

            prompt_index = completion_index // kwargs["n"]
            prompt_ids = stage_1_output["prompt_ids"][prompt_index]
            stage_2_prompt_ids = prompt_ids + completion_ids

            stage_2_prompts.append(stage_2_prompt_ids)

        stage_2_kwargs = copy.deepcopy(kwargs)
        stage_2_kwargs["n"] = 1
        stage_2_kwargs["max_tokens"] = max_tokens - self.thinking_budget

        stage_2_output = self.generate(stage_2_prompts, **stage_2_kwargs)

        # Combine the outputs from stage 1 and stage 2
        for (
            stage_1_completion_ids,
            stage_1_logprobs,
            stage_2_completion_ids,
            stage_2_logprobs
        ) in zip(
            stage_1_output["completion_ids"],
            stage_1_output["logprobs"],
            stage_2_output["completion_ids"],
            stage_2_output["logprobs"],
            strict=True
        ):
            stage_1_completion_ids += stage_2_completion_ids
            stage_1_logprobs += stage_2_logprobs

        return stage_1_output

    def _colocate_call(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> ColocateGenerateOutput:
        assert not sampling_params.stop
        assert not sampling_params.stop_token_ids
        max_tokens = sampling_params.max_tokens

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_params = sampling_params.clone()
        stage_1_params.max_tokens = (
            self.thinking_budget - len(self.stop_thinking_ids)
        )
        stage_1_params.stop = ["</think>\n\n"]
        stage_1_params.include_stop_str_in_output = True

        stage_1_requests = self.generate(prompts, stage_1_params, **kwargs)

        # Stage 2: Generate the final answer
        stage_1_indices = []
        stage_2_prompts = []

        for request_index, request in enumerate(stage_1_requests):
            for completion_index, completion in enumerate(request.outputs):
                assert self.eos_token_id not in completion.token_ids

                if self.eoth_token_id not in completion.token_ids:
                    # Stop the thinking by inserting `self.stop_thinking_ids`
                    completion.text += STOP_THINKING_PROMPT
                    completion.token_ids += self.stop_thinking_ids

                    if sampling_params.logprobs is not None:
                        completion.logprobs += [
                            {s: Logprob(logprob=0.0)}
                            for s in self.stop_thinking_ids
                        ]

                prompt_ids = request.prompt_token_ids + completion.token_ids

                stage_1_indices.append((request_index, completion_index))
                stage_2_prompts.append({"prompt_token_ids": prompt_ids})

        stage_2_params = sampling_params.clone()
        stage_2_params.n = 1
        stage_2_params.max_tokens = max_tokens - self.thinking_budget

        stage_2_requests = self.generate(
            stage_2_prompts, stage_2_params, **kwargs
        )

        # Combine the outputs from stage 1 and stage 2
        for (request_index, completion_index), stage_2_request in zip(
            stage_1_indices, stage_2_requests, strict=True
        ):
            request = stage_1_requests[request_index]
            completion = request.outputs[completion_index]
            stage_2_completion = stage_2_request.outputs[0]

            completion.text += stage_2_completion.text
            completion.token_ids += stage_2_completion.token_ids

            if sampling_params.logprobs is not None:
                completion.cumulative_logprob += (
                    stage_2_completion.cumulative_logprob
                )
                completion.logprobs += stage_2_completion.logprobs

            completion.finish_reason = stage_2_completion.finish_reason

        return stage_1_requests
