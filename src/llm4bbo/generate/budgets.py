import copy
from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizerBase
from trl.generation.vllm_client import VLLMClient

from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob

from .typings import ColocateOutput, ServerOutput


STOP_THINKING_PROMPT = """

Considering the limited time by the user, \
I have to either make a tool call or give the final answer \
based on the thinking directly now.
</think>

"""


# TODO: API model collect SFT data


class GenerateWithBudgets:
    def __init__(
        self,
        generate_func: Callable[..., ColocateOutput | ServerOutput],
        tokenizer: PreTrainedTokenizerBase,
        thinking_budget: int,
        answer_budget: int
    ) -> None:
        self.generate_func = generate_func
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        self.answer_budget = answer_budget

        self.eos_token_id = self.tokenizer.eos_token_id
        assert self.eos_token_id is not None

        self.eoth_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        assert self.eoth_token_id != self.tokenizer.unk_token_id

        self.stop_thinking_ids = self.tokenizer.encode(
            STOP_THINKING_PROMPT, add_special_tokens=False
        )
        self.stop_thinking_tokens = [
            self.tokenizer.decode(i) for i in self.stop_thinking_ids
        ]
        assert self.thinking_budget > len(self.stop_thinking_ids)

    def __call__(self, *args: Any, **kwargs: Any) -> ColocateOutput | ServerOutput:
        if isinstance(self.generate_func.__self__, LLM):
            return self._colocate_call(*args, **kwargs)
        elif isinstance(self.generate_func.__self__, VLLMClient):
            return self._server_call(*args, **kwargs)
        else:
            raise TypeError(
                "Invalid type for `self.generate_func.__self__`: "
                f"{type(self.generate_func.__self__)}"
            )

    def _colocate_call(self,
        prompts: list[dict[str, list[int]]],
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> ColocateOutput:
        assert not sampling_params.stop
        assert not sampling_params.stop_token_ids
        assert sampling_params.max_tokens >= self.thinking_budget + self.answer_budget

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_params = sampling_params.clone()
        stage_1_params.stop = ["</think>\n\n"]
        stage_1_params.max_tokens = self.thinking_budget - len(self.stop_thinking_ids)
        stage_1_params.include_stop_str_in_output = True

        stage_1_requests = self.generate_func(prompts, stage_1_params, **kwargs)

        # Stage 2: Generate after thinking (skip when stage 1 already hits EOS)
        stage_2_prompts = []

        for request in stage_1_requests:
            for completion in request.outputs:
                if self.eos_token_id in completion.token_ids:
                    # Model outputs EOS before </think>
                    continue

                if self.eoth_token_id not in completion.token_ids:
                    # Forcibly stop thinking
                    completion.text += STOP_THINKING_PROMPT
                    completion.token_ids += self.stop_thinking_ids

                    if completion.logprobs is not None:
                        completion.logprobs += [
                            {i: Logprob(logprob=0.0, decoded_token=t)}
                            for i, t in zip(
                                self.stop_thinking_ids,
                                self.stop_thinking_tokens,
                                strict=True,
                            )
                        ]

                prompt_ids = request.prompt_token_ids + completion.token_ids
                stage_2_prompts.append({"prompt_token_ids": prompt_ids})

        stage_2_params = sampling_params.clone()
        stage_2_params.n = 1
        stage_2_params.max_tokens = self.answer_budget

        stage_2_requests = []

        if stage_2_prompts:
            stage_2_requests = self.generate_func(
                stage_2_prompts, stage_2_params, **kwargs
            )

        # Append the outputs from stage 2 to stage 1
        stage_2_iter = iter(stage_2_requests)

        for request in stage_1_requests:
            for completion in request.outputs:
                if self.eos_token_id in completion.token_ids:
                    continue

                stage_2_completion = next(stage_2_iter).outputs[0]

                completion.text += stage_2_completion.text
                completion.token_ids += stage_2_completion.token_ids

                if completion.cumulative_logprob is not None:
                    completion.cumulative_logprob += (
                        stage_2_completion.cumulative_logprob
                    )

                if completion.logprobs is not None:
                    completion.logprobs += stage_2_completion.logprobs

                completion.finish_reason = stage_2_completion.finish_reason
                completion.stop_reason = stage_2_completion.stop_reason

        return stage_1_requests

    def _server_call(
        self,
        prompts: list[list[int]],
        **kwargs: Any
    ) -> ServerOutput:
        kwargs["generation_kwargs"] = kwargs.get("generation_kwargs") or {}

        assert kwargs["max_tokens"] >= self.thinking_budget + self.answer_budget
        assert "stop" not in kwargs["generation_kwargs"]
        assert "stop_token_ids" not in kwargs["generation_kwargs"]
        assert "max_tokens" not in kwargs["generation_kwargs"]

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        stage_1_kwargs = copy.deepcopy(kwargs)
        stage_1_kwargs["max_tokens"] = (
            self.thinking_budget - len(self.stop_thinking_ids)
        )
        stage_1_kwargs["generation_kwargs"]["stop"] = ["</think>\n\n"]

        stage_1_output = self.generate_func(prompts, **stage_1_kwargs)

        # Stage 2: Generate after thinking (skip when stage 1 already hits EOS)
        stage_2_prompts = []

        for (
            completion_index, (completion_ids, logprobs, logprob_token_ids)
        ) in enumerate(
            zip(
                stage_1_output["completion_ids"],
                stage_1_output["logprobs"],
                stage_1_output["logprob_token_ids"],
                strict=True
            )
        ):
            if self.eos_token_id in completion_ids:
                # Model outputs EOS before </think>
                continue

            if self.eoth_token_id not in completion_ids:
                # Forcibly stop thinking
                completion_ids += self.stop_thinking_ids
                logprobs += [[0.0]] * len(self.stop_thinking_ids)
                logprob_token_ids += [[i] for i in self.stop_thinking_ids]

            prompt_index = completion_index // kwargs["n"]
            prompt_ids = stage_1_output["prompt_ids"][prompt_index]
            stage_2_prompts.append(prompt_ids + completion_ids)

        stage_2_kwargs = copy.deepcopy(kwargs)
        stage_2_kwargs["n"] = 1
        stage_2_kwargs["max_tokens"] = self.answer_budget

        stage_2_output = {"completion_ids": [], "logprobs": [], "logprob_token_ids": []}

        if stage_2_prompts:
            stage_2_output = self.generate_func(stage_2_prompts, **stage_2_kwargs)

        # Append the outputs from stage 2 to stage 1
        stage_2_iter = iter(
            zip(
                stage_2_output["completion_ids"],
                stage_2_output["logprobs"],
                stage_2_output["logprob_token_ids"],
                strict=True
            )
        )

        for completion_ids, logprobs, logprob_token_ids in zip(
            stage_1_output["completion_ids"],
            stage_1_output["logprobs"],
            stage_1_output["logprob_token_ids"],
            strict=True
        ):
            if self.eos_token_id in completion_ids:
                continue

            stage_2_completion_ids, stage_2_logprobs, stage_2_logprob_token_ids = (
                next(stage_2_iter)
            )

            completion_ids += stage_2_completion_ids
            logprobs += stage_2_logprobs
            logprob_token_ids += stage_2_logprob_token_ids

        return stage_1_output
