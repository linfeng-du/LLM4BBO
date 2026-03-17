import os
os.environ["USE_TF"] = "0"

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import llm4bbo.patches, design_bench

import gpytorch
import torch

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin
)
from transformers.pipelines.text_generation import ChatType
from trl import GRPOConfig, GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.generation.vllm_generation import sanitize_logprob

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from llm4bbo.dataset import create_parse_fn, build_dataset
from llm4bbo.reward import ImprovedExactGPModel
from llm4bbo.trainer.hf.evaluate import evaluate
from llm4bbo.trainer.hf.thinking_budget import ThinkingBudgetVLLMGeneration
from llm4bbo.trainer.hf.utils import get_best_model, update_config


@hydra.main(
    config_path="config", config_name="online_rl_trainer", version_base=None
)
def main(cfg: DictConfig) -> None:
    update_config(cfg)
    main_online_rl(cfg)


def main_online_rl(cfg: DictConfig) -> None:
    wandb.init(**OmegaConf.to_container(cfg.wandb_init, resolve=True))

    dataset = build_dataset(**cfg.build_dataset)

    if cfg.init_from == "base":
        model = cfg.llm.model
    else:
        model = get_best_model(cfg)

    trainer = ThinkingBudgetGRPOTrainer(
        model,
        thinking_budget=cfg.thinking_budget,
        eoth_token=cfg.llm.eoth_token,
        reward_funcs=[create_gaussian_process_reward(cfg.task_name)],
        args=GRPOConfig(
            **OmegaConf.to_container(cfg.grpo_config, resolve=True)
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    evaluate(cfg)


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


def create_gaussian_process_reward(
    task_name: str
) -> Callable[[list[ChatType], list[float]], list[float]]:
    task = design_bench.make(task_name)
    parse_fn = create_parse_fn(task_name)

    checkpoint_path = Path("data") / "gp_models" / f"gp_model_{task_name}_0.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = (
        ImprovedExactGPModel(
            checkpoint["train_x"].to(device),
            checkpoint["train_y"].to(device),
            likelihood,
            checkpoint["rbf_only"]
        )
        .to(device)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    model.eval()
    likelihood.eval()

    def gaussian_process_reward(
        completions: list[ChatType],
        best_reference_score: list[float],
        **kwargs: Any
    ) -> list[float]:
        x_pred = parse_fn([c[0]["content"] for c in completions])

        if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
            x_pred = task.to_logits(x_pred).reshape(len(x_pred), -1)

        x_pred = torch.from_numpy(x_pred).to(device, dtype=torch.float32)
        best_reference_score = torch.tensor(best_reference_score, device=device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = likelihood(model(x_pred))
            y_pred = distribution.mean.squeeze(dim=-1)

        return (y_pred - best_reference_score).tolist()

    return gaussian_process_reward


if __name__ == "__main__":
    main()
