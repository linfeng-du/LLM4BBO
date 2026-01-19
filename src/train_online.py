import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Prevent Transformers from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig

import gpytorch
import torch

from peft import AutoPeftModelForCausalLM
from transformers import pipeline
from transformers.pipelines.text_generation import ChatType
from trl import GRPOConfig, GRPOTrainer

import patches
import design_bench

from dataset import build_rl_dataset
from gaussian_process_improved import ImprovedExactGPModel
from prompt import create_parse_fn
from test import test


@hydra.main(
    config_path="../conf",
    config_name="train_online",
    version_base=None
)
def train_online(cfg: DictConfig):
    model_dirs = list(
        (Path("outputs") / cfg.run_name)
        .parent.parent.glob("train_offline/checkpoint-*")
    )
    assert len(model_dirs) == 1
    model_dir = model_dirs[0]

    dataset = build_rl_dataset(mode="online", **cfg.dataset)

    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    trainer = GRPOTrainer(
        AutoPeftModelForCausalLM.from_pretrained(model_dir, is_trainable=True),
        reward_funcs=[create_gaussian_process_reward(cfg.task_name)],
        args=GRPOConfig(**cfg.trainer_args),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()

    pipe = pipeline(
        task="text-generation",
        model=trainer.model,
        tokenizer=trainer.processing_class,
        **cfg.llm.hf_generation_kwargs
    )
    test(pipe, cfg)    


def create_gaussian_process_reward(
    task_name: str,
    device: str = "cuda"
) -> Callable[[list[ChatType], list[float], dict[str, Any]], list[float]]:
    task = design_bench.make(task_name)
    parse_fn = create_parse_fn(task_name)

    checkpoint_path = Path("data") / f"gp_model_{task_name}_0.pt"
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
        **kwargs: dict[str, Any]
    ) -> list[float]:
        completions = [c[0]["content"] for c in completions]
        x_design = parse_fn(completions)

        if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
            x_design = task.to_logits(x_design).reshape(len(x_design), -1)

        x_design = torch.from_numpy(x_design).float().to(device)
        best_reference_score = torch.tensor(
            best_reference_score, dtype=torch.float, device=device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = likelihood(model(x_design))
            y_design = distribution.mean.squeeze(dim=-1)

        return (y_design - best_reference_score).tolist()

    return gaussian_process_reward


if __name__ == "__main__":
    train_online()
