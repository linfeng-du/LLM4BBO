import os
os.environ["USE_TF"] = "0"

import gc
import multiprocessing as mp
from collections.abc import Callable
from importlib import resources
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import llm4bbo.patches, design_bench

import gpytorch
import torch

from transformers import AutoTokenizer
from transformers.pipelines.text_generation import ChatType
from trl import GRPOConfig, GRPOTrainer

from llm4bbo.dataset import build_dataset, create_parse_fn
from llm4bbo.reward.gaussian_process_improved import ImprovedExactGPModel
from llm4bbo.trainer.evaluate import evaluate
from llm4bbo.trainer.thinking_budget import ThinkingBudgetVLLMGenerate
from llm4bbo.trainer.utils import get_best_model, update_config


@hydra.main(config_path="config", config_name="online_rl_trainer", version_base=None)
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

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.model, padding_side="left", truncation_side="left"
    )

    trainer = GRPOTrainer(
        model,
        reward_funcs=[create_gaussian_process_reward(cfg.task_name)],
        args=GRPOConfig(**OmegaConf.to_container(cfg.grpo_config, resolve=True)),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        rollout_func=thinking_budget_rollout_func
    )

    if cfg.grpo_config.vllm_mode == "server":
        trainer.vllm_generation.vllm_client.generate = ThinkingBudgetVLLMGenerate(
            trainer.vllm_generation.vllm_client.generate,
            tokenizer,
            cfg.thinking_budget,
            cfg.answer_budget
        )
    elif cfg.grpo_config.vllm_mode == "colocate":
        trainer.vllm_generation.llm.generate = ThinkingBudgetVLLMGenerate(
            trainer.vllm_generation.llm.generate,
            tokenizer,
            cfg.thinking_budget,
            cfg.answer_budget
        )

    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if cfg.grpo_config.vllm_mode == "colocate":
        ctx = mp.get_context(method="spawn")

        OmegaConf.resolve(cfg)
        results_queue = ctx.Queue()

        p = ctx.Process(target=evaluate, args=(cfg, results_queue))
        p.start()
        results = results_queue.get()
        p.join()

        wandb.summary.update(results["evaluate"])
        wandb.summary["evaluate/best_conversations"] = wandb.Table(**results["table"])
    else:
        evaluate(cfg)


def thinking_budget_rollout_func(
    prompts: list[ChatType],
    trainer: GRPOTrainer
) -> dict[str, list[list[int]] | list[list[float]]]:
    # https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py#L1572
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)
    completion_ids, logprobs, _ = trainer._generate_single_turn(
        prompt_ids, images, multimodal_fields
    )

    # Mask the stop thinking tokens in `completion_ids`
    env_mask = [[1] * len(ids) for ids in completion_ids]

    if trainer.vllm_generation.mode == "server":
        generate = trainer.vllm_generation.vllm_client.generate
    elif trainer.vllm_generation.mode == "colocate":
        generate = trainer.vllm_generation.llm.generate

    for ids, mask in zip(completion_ids, env_mask, strict=True):
        if generate.eoth_token_id not in ids:
            continue

        eoth_index = ids.index(generate.eoth_token_id)
        eoth_offset = generate.stop_thinking_ids.index(generate.eoth_token_id)
        mask_start = eoth_index - eoth_offset
        mask_end = mask_start + len(generate.stop_thinking_ids)

        if ids[mask_start:mask_end] == generate.stop_thinking_ids:
            mask[mask_start:mask_end] = [0] * len(generate.stop_thinking_ids)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask
    }


def create_gaussian_process_reward(
    task_name: str
) -> Callable[[list[ChatType], list[float]], list[float]]:
    task = design_bench.make(task_name)
    parse_fn = create_parse_fn(task_name)

    checkpoint_path = (
        resources.files("llm4bbo") / "data" / "gp_models" / f"gp_model_{task_name}_0.pt"
    )
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

        device = model.train_inputs[0].device
        x_pred = torch.from_numpy(x_pred).to(device, torch.float32)
        best_reference_score = torch.tensor(best_reference_score, device=device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = likelihood(model(x_pred))
            y_pred = distribution.mean.squeeze(dim=-1)

        return (y_pred - best_reference_score).tolist()

    return gaussian_process_reward


if __name__ == "__main__":
    main()
