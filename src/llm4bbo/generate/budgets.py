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


class GenerateWithBudgets:
    def __init__(
        self,
        generate_func: Callable,
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

    def _colocate_call(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> ColocateOutput:
        assert not sampling_params.stop
        assert not sampling_params.stop_token_ids
        assert sampling_params.max_tokens >= self.thinking_budget + self.answer_budget

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        params = sampling_params.clone()
        params.stop = ["</think>\n\n"]
        params.max_tokens = self.thinking_budget - len(self.stop_thinking_ids)
        params.include_stop_str_in_output = True

        requests = self.generate_func(prompts, params, **kwargs)

        # Process stage 1 outputs
        new_prompts = []
        completions = []

        for request in requests:
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
                            {i: Logprob(0.0, decoded_token=t)}
                            for i, t in zip(
                                self.stop_thinking_ids,
                                self.stop_thinking_tokens,
                                strict=True
                            )
                        ]

                new_prompt_ids = request.prompt_token_ids + completion.token_ids
                new_prompts.append({"prompt_token_ids": new_prompt_ids})
                completions.append(completion)

        # Stage 2: Generate after thinking (skip when stage 1 already hits EOS)
        params = sampling_params.clone()
        params.n = 1
        params.max_tokens = self.answer_budget

        new_requests = []

        if new_prompts:
            new_requests = self.generate_func(new_prompts, params, **kwargs)

        # Append stage 2 outputs to stage 1
        for completion, new_request in zip(completions, new_requests, strict=True):
            new_completion = new_request.outputs[0]

            completion.text += new_completion.text
            completion.token_ids += new_completion.token_ids
            completion.finish_reason = new_completion.finish_reason
            completion.stop_reason = new_completion.stop_reason

            if completion.cumulative_logprob is not None:
                completion.cumulative_logprob += new_completion.cumulative_logprob

            if completion.logprobs is not None:
                completion.logprobs += new_completion.logprobs

        return requests

    def _server_call(
        self,
        prompts: list[list[int]],
        n: int = 1,
        **kwargs: Any
    ) -> ServerOutput:
        kwargs["generation_kwargs"] = kwargs.get("generation_kwargs") or {}

        assert kwargs["max_tokens"] >= self.thinking_budget + self.answer_budget
        assert "stop" not in kwargs["generation_kwargs"]
        assert "stop_token_ids" not in kwargs["generation_kwargs"]
        assert "max_tokens" not in kwargs["generation_kwargs"]

        # Stage 1: Generate thinking up to `self.thinking_budget` tokens
        params = copy.deepcopy(kwargs)
        params["n"] = n
        params["max_tokens"] = self.thinking_budget - len(self.stop_thinking_ids)
        params["generation_kwargs"]["stop"] = ["</think>\n\n"]

        outputs = self.generate_func(prompts, **params)

        # Process stage 1 outputs
        new_prompts = []
        indices = []

        for index in range(len(outputs["completion_ids"])):
            if self.eos_token_id in outputs["completion_ids"][index]:
                # Model outputs EOS before </think>
                continue

            if self.eoth_token_id not in outputs["completion_ids"][index]:
                # Forcibly stop thinking
                outputs["completion_ids"][index] += self.stop_thinking_ids

                if outputs["logprobs"] is not None:
                    outputs["logprobs"][index] += [
                        [0.0] for _ in range(len(self.stop_thinking_ids))
                    ]

                if outputs["logprob_token_ids"] is not None:
                    outputs["logprob_token_ids"][index] += [
                        [i] for i in self.stop_thinking_ids
                    ]

            new_prompt_ids = (
                outputs["prompt_ids"][index // n] + outputs["completion_ids"][index]
            )
            new_prompts.append(new_prompt_ids)
            indices.append(index)

        # Stage 2: Generate after thinking (skip when stage 1 already hits EOS)
        params = copy.deepcopy(kwargs)
        params["n"] = 1
        params["max_tokens"] = self.answer_budget

        new_outputs = {"completion_ids": [], "logprobs": [], "logprob_token_ids": []}

        if new_prompts:
            new_outputs = self.generate_func(new_prompts, **params)

        # Append stage 2 outputs to stage 1
        for new_index, index in enumerate(indices):
            outputs["completion_ids"][index] += new_outputs["completion_ids"][new_index]

            if outputs["logprobs"] is not None:
                outputs["logprobs"][index] += new_outputs["logprobs"][new_index]

            if outputs["logprob_token_ids"] is not None:
                outputs["logprob_token_ids"][index] += (
                    new_outputs["logprob_token_ids"][new_index]
                )

        return outputs
