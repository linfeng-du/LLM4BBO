from typing import Any

from transformers import PreTrainedTokenizerBase
from trl.generation.vllm_client import VLLMClient

from vllm import LLM, RequestOutput, SamplingParams
from vllm.logprobs import Logprob


EARLY_STOPPING_PROMPT = """

Considering the limited time by the user, \
I have to give the solution based on the thinking directly now.
{eoth_token}

"""


class ThinkingBudgetVLLMGeneration:
    def __init__(
        self,
        vllm_client_or_llm: VLLMClient | LLM,
        tokenizer: PreTrainedTokenizerBase,
        thinking_budget: int,
        eoth_token: str
    ) -> None:
        if isinstance(vllm_client_or_llm, VLLMClient):
            self.vllm_mode = "server"
            self.vllm_client = vllm_client_or_llm
        elif isinstance(vllm_client_or_llm, LLM):
            self.vllm_mode = "colocate"
            self.llm = vllm_client_or_llm

        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget

        self.eos_token_id = self.tokenizer.eos_token_id
        self.eoth_token_id = self.tokenizer.convert_tokens_to_ids(eoth_token)
        assert self.eos_token_id is not None
        assert self.eoth_token_id != self.tokenizer.unk_token_id

        self.early_stopping_prompt = EARLY_STOPPING_PROMPT.format(
            eoth_token=eoth_token
        )
        self.early_stopping_ids = self.tokenizer.encode(
            self.early_stopping_prompt, add_special_tokens=False
        )

    def __call__(self, *args: Any, **kwargs: Any) -> (
        dict[str, list[list[int]] | list[list[float]]] | list[RequestOutput]
    ):
        if self.vllm_mode == "server":
            return self._server_call(*args, **kwargs)
        elif self.vllm_mode == "colocate":
            return self._colocate_call(*args, **kwargs)

    def _server_call(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any]
    ) -> dict[str, list[list[int]] | list[list[float]]]:
        max_tokens = sampling_params["max_tokens"]

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_params = sampling_params.copy()
        stage_1_params["max_tokens"] = self.thinking_budget
        stage_1_output = self.vllm_client.generate(prompts, **stage_1_params)

        # Stage 2: Generate the final answer
        stage_1_indices = []
        stage_2_prompts = []
        stage_2_max_tokens = []

        for completion_index, completion_ids in enumerate(
            stage_1_output["completion_ids"]
        ):
            if self.eos_token_id in completion_ids:
                # Completion has already finished
                continue

            prompt_index = completion_index // sampling_params["n"]
            stage_1_prompt_ids = stage_1_output["prompt_ids"][prompt_index]
            prompt_ids = stage_1_prompt_ids + completion_ids

            if self.eoth_token_id not in prompt_ids:
                # End the thinking by inserting `self.early_stopping_ids`
                prompt_ids += self.early_stopping_ids

            prompt = self.tokenizer.decode(prompt_ids)

            remaining_max_tokens = (
                max_tokens + len(stage_1_prompt_ids) - len(prompt_ids)
            )

            stage_1_indices.append(completion_index)
            stage_2_prompts.append(prompt)
            stage_2_max_tokens.append(remaining_max_tokens)

        if not stage_2_prompts:
            return stage_1_output

        # Batch stage 2 with the largest remaining max tokens
        # to avoid many concurrent requests to the vLLM server.
        stage_2_params = sampling_params.copy()
        stage_2_params["n"] = 1
        stage_2_params["max_tokens"] = max(stage_2_max_tokens)
        stage_2_output = self.vllm_client.generate(
            stage_2_prompts, **stage_2_params
        )

        # Combine the outputs from stage 1 and stage 2
        for stage_2_index, stage_1_index in enumerate(stage_1_indices):
            stage_1_output["completion_ids"][stage_1_index] += (
                stage_2_output["completion_ids"][stage_2_index]
            )
            stage_1_output["logprobs"][stage_1_index] += (
                stage_2_output["logprobs"][stage_2_index]
            )

        return stage_1_output

    def _colocate_call(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True
    ) -> list[RequestOutput]:
        assert not sampling_params.stop
        assert not sampling_params.stop_token_ids
        max_tokens = sampling_params.max_tokens

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_params = sampling_params.clone()
        stage_1_params.max_tokens = self.thinking_budget
        stage_1_requests = self.llm.generate(
            prompts, stage_1_params, use_tqdm=use_tqdm
        )

        # Stage 2: Generate the final answer
        stage_1_indices = []
        stage_2_prompts = []
        stage_2_params = []

        for request_index, request in enumerate(stage_1_requests):
            for completion_index, completion in enumerate(request.outputs):
                if self.eos_token_id in completion.token_ids:
                    # Completion has already finished
                    continue

                prompt_ids = request.prompt_token_ids + completion.token_ids

                if self.eoth_token_id not in prompt_ids:
                    # End the thinking by inserting `self.early_stopping_ids`
                    prompt_ids += self.early_stopping_ids

                params = sampling_params.clone()
                params.n = 1
                params.max_tokens = (
                    max_tokens
                    + len(request.prompt_token_ids)
                    - len(prompt_ids)
                )

                stage_1_indices.append((request_index, completion_index))
                stage_2_prompts.append({"prompt_token_ids": prompt_ids})
                stage_2_params.append(params)

        if not stage_2_prompts:
            return stage_1_requests

        stage_2_requests = self.llm.generate(
            stage_2_prompts, stage_2_params, use_tqdm=use_tqdm
        )

        # Combine the outputs from stage 1 and stage 2
        for stage_1_index, stage_2_request in zip(
            stage_1_indices, stage_2_requests, strict=True
        ):
            request = stage_1_requests[stage_1_index[0]]
            completion = request.outputs[stage_1_index[1]]
            stage_2_completion = stage_2_request.outputs[0]

            completion.text += (
                self.early_stopping_prompt + stage_2_completion.text
            )
            completion.token_ids += (
                self.early_stopping_ids + stage_2_completion.token_ids
            )

            if sampling_params.logprobs is not None:
                completion.cumulative_logprob += (
                    stage_2_completion.cumulative_logprob
                )
                completion.logprobs += (
                    [
                        {i: Logprob(logprob=0.0)}
                        for i in self.early_stopping_ids
                    ]
                    + stage_2_completion.logprobs
                )

            completion.finish_reason = stage_2_completion.finish_reason

        return stage_1_requests
