import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Prevent Transformers from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import gpytorch
import torch

from transformers.pipelines.text_generation import ChatType
from trl import GRPOConfig

import patches, design_bench

from dataset import build_dataset
from gaussian_process_improved import ImprovedExactGPModel
from prompt import create_parse_fn
from resolver import register_resolvers
from thinking_budget import ThinkingBudgetGRPOTrainer


register_resolvers()


@hydra.main(config_path="../conf", config_name="online_rl", version_base=None)
def train_online_rl(cfg: DictConfig):
    dataset = build_dataset(stage="online_rl", **cfg.build_dataset)

    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    trainer = ThinkingBudgetGRPOTrainer(
        cfg.base_model,
        thinking_budget=cfg.online_rl.thinking_budget,
        eoth_token=cfg.llm.eoth_token,
        reward_funcs=[create_gaussian_process_reward(cfg.task_name)],
        args=GRPOConfig(
            **OmegaConf.to_container(cfg.training_arguments, resolve=True)
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()


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
        best_reference_score = torch.tensor(
            best_reference_score, device=device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = likelihood(model(x_pred))
            y_pred = distribution.mean.squeeze(dim=-1)

        return (y_pred - best_reference_score).tolist()

    return gaussian_process_reward


if __name__ == "__main__":
    train_online_rl()
