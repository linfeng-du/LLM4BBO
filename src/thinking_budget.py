import asyncio
from typing import Any

import torch

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin
)
from transformers.pipelines.text_generation import ChatType

from trl import GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.generation.vllm_client import VLLMClient
from trl.generation.vllm_generation import sanitize_logprob

from vllm import LLM, RequestOutput, SamplingParams
from vllm.logprobs import Logprob
from vllm.sampling_params import StructuredOutputsParams


class ThinkingBudgetGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: str | PreTrainedModel,
        thinking_budget: int,
        eoth_token: str,
        **kwargs: Any
    ):
        args = kwargs.get("args", None)
        assert args is not None and args.use_vllm
        assert kwargs.get("tools", None) is None
        assert kwargs.get("rollout_func", None) is None

        if args.vllm_mode == "server":
            rollout_func = _server_rollout_func
        elif args.vllm_mode == "colocate":
            rollout_func = _colocate_rollout_func

        super().__init__(model, rollout_func=rollout_func, **kwargs)
        # Set `tools` to enter `_tool_call_loop`
        self.tools = True

        if self.args.vllm_mode == "server":
            vllm_client_or_llm = self.vllm_generation.vllm_client
        elif self.args.vllm_mode == "colocate":
            vllm_client_or_llm = self.vllm_generation.llm

        if isinstance(self.processing_class, ProcessorMixin):
            tokenizer = self.processing_class.tokenizer
        elif isinstance(self.processing_class, PreTrainedTokenizerBase):
            tokenizer = self.processing_class

        self.thinking_budget_vllm_generation = ThinkingBudgetVLLMGeneration(
            vllm_client_or_llm, tokenizer, thinking_budget, eoth_token
        )

        self.generation_kwargs = self.args.generation_kwargs
        self.structured_outputs_regex = self.args.vllm_structured_outputs_regex
        self.tensor_parallel_size = self.args.vllm_tensor_parallel_size

        self.thinking_budget = thinking_budget
        self.eoth_token_id = tokenizer.convert_tokens_to_ids(eoth_token)
        assert self.eoth_token_id != tokenizer.unk_token_id

    def _tool_call_loop(
        self,
        prompts: list[ChatType],
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        completions: list[ChatType],
        logprobs: list[list[float]]
    ) -> tuple[
        list[list[int]],
        list[ChatType],
        list[list[int]],
        list[list[float]],
        int,
        int
    ]:
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0

        for ids, mask in zip(completion_ids, tool_mask, strict=True):
            if self.eoth_token_id not in ids:
                # Thinking should always end with `eoth_token`
                tool_failure_count += 1
                continue

            eoth_index = ids.index(self.eoth_token_id)

            if eoth_index >= self.thinking_budget:
                mask[self.thinking_budget : eoth_index + 1] = (
                    [0] * (eoth_index - self.thinking_budget + 1)
                )
                tool_call_count += 1

        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            tool_call_count,
            tool_failure_count
        )


def _server_rollout_func(
    prompts: list[str],
    self: ThinkingBudgetGRPOTrainer
) -> dict[str, list[list[int]] | list[list[float]]]:
    # https://github.com/huggingface/trl/blob/v0.28.0/trl/generation/vllm_generation.py#L566
    sampling_params = {
        "n": self.num_generations,
        "repetition_penalty": self.repetition_penalty,
        "temperature": self.temperature,
        "top_p": self.top_p,
        "top_k": self.top_k,
        "min_p": 0.0 if self.min_p is None else self.min_p,
        "max_tokens": self.max_completion_length,
        "structured_outputs_regex": self.structured_outputs_regex,
        "generation_kwargs": self.generation_kwargs,
    }
    return self.thinking_budget_vllm_generation(prompts, sampling_params)


def _colocate_rollout_func(
    prompts: list[str],
    self: ThinkingBudgetGRPOTrainer
) -> dict[str, list[list[int]] | list[list[float]]]:
    # https://github.com/huggingface/trl/blob/v0.28.0/trl/generation/vllm_generation.py#L627
    structured_outputs_key = "structured_outputs"
    if self.structured_outputs_regex:
        structured_outputs = StructuredOutputsParams(regex=self.structured_outputs_regex)
    else:
        structured_outputs = None

    generation_kwargs = {
        "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
        "repetition_penalty": self.repetition_penalty,
        "temperature": self.temperature,
        "top_p": self.top_p,
        "top_k": self.top_k,
        "min_p": 0.0 if self.min_p is None else self.min_p,
        "max_tokens": self.max_completion_length,
        "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
    }
    generation_kwargs[structured_outputs_key] = structured_outputs
    generation_kwargs.update(self.generation_kwargs)
    sampling_params = SamplingParams(**generation_kwargs)

    if self.tensor_parallel_size > 1:
        # Gather prompts from all ranks in the TP group and flatten.
        # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
        orig_size = len(prompts)
        gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
        all_prompts = [p for sublist in gathered_prompts for p in sublist]
    else:
        all_prompts = prompts

    with profiling_context(self, "vLLM.generate"):
        all_outputs = self.thinking_budget_vllm_generation(
            all_prompts, sampling_params, use_tqdm=False
        )

    all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
    all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
    all_logprobs = [
        [sanitize_logprob(next(iter(lp.values()))) for lp in output.logprobs]
        for outputs in all_outputs
        for output in outputs.outputs
    ]

    if self.tensor_parallel_size > 1:
        # Slice completions for this rank within its TP group.
        # Each rank generates all outputs — we keep only our share.
        local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
        tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
        prompt_ids = all_prompt_ids[tp_slice]
        completion_ids = all_completion_ids[tp_slice]
        logprobs = all_logprobs[tp_slice]
    else:
        prompt_ids = all_prompt_ids
        completion_ids = all_completion_ids
        logprobs = all_logprobs

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs
    }


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

        # Stage 2: Generate final answer
        stage_1_indices = []
        stage_2_prompts = []
        stage_2_params = []

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

            params = sampling_params.copy()
            params["n"] = 1
            params["max_tokens"] = (
                max_tokens + len(stage_1_prompt_ids) - len(prompt_ids)
            )

            stage_1_indices.append(completion_index)
            stage_2_prompts.append(prompt)
            stage_2_params.append(params)

        if not stage_2_prompts:
            return stage_1_output

        stage_2_outputs = asyncio.run(
            self._async_generate(stage_2_prompts, stage_2_params)
        )

        # Combine the outputs from stage 1 and stage 2
        for stage_1_index, stage_2_output in zip(
            stage_1_indices, stage_2_outputs, strict=True
        ):
            stage_1_output["completion_ids"][stage_1_index] += (
                stage_2_output["completion_ids"][0]
            )
            stage_1_output["logprobs"][stage_1_index] += (
                stage_2_output["logprobs"][0]
            )

        return stage_1_output

    async def _async_generate(
        self,
        prompts: list[str],
        sampling_params: list[dict[str, Any]]
    ) -> list[dict[str, list[list[int]] | list[list[float]]]]:
        tasks = [
            asyncio.to_thread(self.vllm_client.generate, [p], **s)
            for p, s in zip(prompts, sampling_params, strict=True)
        ]
        return await asyncio.gather(*tasks)

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

        # Stage 2: Generate final answer
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
