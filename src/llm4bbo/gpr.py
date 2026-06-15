from collections.abc import Callable
from importlib import resources
from typing import Any

import llm4bbo.patches
import design_bench

import numpy as np

import torch
from torch.distributions import Normal
from transformers.pipelines.text_generation import ChatType

from botorch.models import SingleTaskGP
from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

from llm4bbo.dataset import create_parse_fn


def create_tool(task_name: str) -> Callable[[str], dict[str, str]]:
    surrogate = GPRSurrogate(task_name)
    parse_fn = create_parse_fn(task_name)

    def predict_score(x: str) -> tuple[float, float]:
        """
        Predict the score and uncertainty of a design.

        Args:
            x: The design to predict the score and uncertainty for.

        Returns:
            The predicted score and uncertainty.
        """
        print(x)
        x = parse_fn([x])

        with torch.no_grad():
            posterior = surrogate.model.posterior(x.unsqueeze(-2))
            mean = posterior.mean.flatten()
            sigma = posterior.variance.clamp_min(1e-9).sqrt().flatten()

        return mean.item(), sigma.item()

    return predict_score


def create_gpr_reward(task_name: str, function: str) -> (
    Callable[[list[ChatType], list[float]], list[float]]
):
    parse_fn = create_parse_fn(task_name)
    surrogate = GPRSurrogate(task_name)

    def gpr_reward(
        completions: list[ChatType],
        best_f: list[float],
        **kwargs: Any
    ) -> list[float]:
        x = parse_fn([c[0]["content"] for c in completions])
        return surrogate.acquisition(x, best_f, function).tolist()

    return gpr_reward


class GPRSurrogate:

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.task = design_bench.make(task_name)

        ckpt_path = resources.files("llm4bbo") / "assets" / "models" / f"{task_name}.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, device)

        self.model = SingleTaskGP(
            ckpt["train_x"].to(device),
            ckpt["train_y"].to(device),
            likelihood=GaussianLikelihood(),
            covar_module=RBFKernel(lengthscale_constraint=Positive()),
            outcome_transform=None
        )
        self.model.load_state_dict(ckpt["state_dict"])

    def acquisition(
        self,
        x: np.ndarray,
        best_f: list[float],
        function: str
    ) -> torch.Tensor:
        if self.task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
            x = self.task.to_logits(x).reshape(len(x), -1)

        x = torch.from_numpy(x).to(self.model.train_inputs[0])
        best_f = torch.tensor(best_f).to(self.model.train_targets)

        with torch.no_grad():
            posterior = self.model.posterior(x.unsqueeze(-2))
            mean = posterior.mean.flatten()
            sigma = posterior.variance.clamp_min(1e-9).sqrt().flatten()

        u = (mean - best_f) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))

        if function == "EI":
            return sigma * (normal.log_prob(u).exp() + u * normal.cdf(u))
        elif function == "PI":
            return normal.cdf(u)
        else:
            raise ValueError(f"Invalid acquisition function: {function}")
