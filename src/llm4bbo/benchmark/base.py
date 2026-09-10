from __future__ import annotations

__all__ = [
    "BenchmarkTask",
    "make_task",
    "register_tasks"
]

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

import numpy as np
from transformers.pipelines.text_generation import ChatType

from .prompts.common import (
    DESIGN_GENERATION_PROMPT_TEMPLATE,
    TOOL_USE_PROMPT_TEMPLATE,
    TRAJECTORY_GENERATION_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE
)
from .utils import evenly_ranked_indices, parse_categorical, parse_numerical


logger = logging.getLogger(__name__)

_ASSETS_DIR = resources.files("llm4bbo") / "assets"
_NUM_PREDICT_WORKERS = len(os.sched_getaffinity(0))


class BenchmarkTask(ABC):
    benchmark: str
    score_precision: int

    def __init__(
        self,
        task_name: str,
        num_designs: int,
        system_prompt: str,
        x_offline: np.ndarray,
        categories: list[str] | None = None,
        allowed_values: list[int] | None = None
    ) -> None:
        if x_offline.ndim != 2:
            raise ValueError("x_offline must be a 2D array")

        if not 0 < num_designs <= len(x_offline):
            raise ValueError(f"num_designs must be in [1, {len(x_offline)}]")

        if categories is not None and allowed_values is not None:
            raise ValueError("categories and allowed_values must not both be set")

        self.task_name = task_name
        self.design_dim = x_offline.shape[1]
        self.num_designs = num_designs

        self.categories = categories
        self.allowed_values = allowed_values

        self.system_prompt = system_prompt

        self.x_offline = x_offline
        cache_path = self.data_dir / f"{self.task_name}_y_offline.npy"
        self.y_offline = self._cached_parallel_predict(self.x_offline, cache_path)

        self.sample_indices = evenly_ranked_indices(self.y_offline, self.num_designs)
        self.x = self.x_offline[self.sample_indices]
        self.y = self.y_offline[self.sample_indices]

    @property
    def benchmark_dir(self) -> Path:
        return _ASSETS_DIR / self.benchmark

    @property
    def data_dir(self) -> Path:
        return self.benchmark_dir / "data"

    def create_prompt_messages(
        self,
        x_references: np.ndarray,
        y_references: np.ndarray,
        thinking_budget: int,
        max_tool_calls: int | None = None,
        x_target: np.ndarray | None = None
    ) -> ChatType:
        system_prompt_parts = [self.system_prompt]

        if max_tool_calls is not None:
            # Enable tool use
            if max_tool_calls <= 0:
                raise ValueError("max_tool_calls must be positive")

            system_prompt_parts.append(
                TOOL_USE_PROMPT_TEMPLATE.format(max_tool_calls=max_tool_calls)
            )

        if x_target is not None:
            # Generate a complete trajectory ending with the given target design
            system_prompt_parts.append(
                TRAJECTORY_GENERATION_PROMPT_TEMPLATE.format(
                    target=self._render_design(x_target),
                    thinking_budget=thinking_budget
                )
            )
        else:
            system_prompt_parts.append(
                DESIGN_GENERATION_PROMPT_TEMPLATE.format(
                    thinking_budget=thinking_budget
                )
            )

        references = "\n".join(
            f"{self._render_design(x_i)}, "
            f"Score: {round(y_i.item(), self.score_precision)}"
            for x_i, y_i in zip(x_references, y_references, strict=True)
        )

        system_prompt = "\n\n".join(system_prompt_parts)
        user_prompt = USER_PROMPT_TEMPLATE.format(references=references)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def create_completion_messages(self, x_response: np.ndarray) -> ChatType:
        return [{"role": "assistant", "content": self._render_design(x_response)}]

    def evaluate(self, completions: list[str]) -> tuple[np.ndarray, int]:
        if not completions:
            raise ValueError("completions must not be empty")

        if self.categories is not None:
            results = [
                parse_categorical(
                    c,
                    self.design_dim,
                    self.categories,
                    self.x_offline.dtype
                )
                for c in completions
            ]
        else:
            results = [
                parse_numerical(
                    c,
                    self.design_dim,
                    self.allowed_values,
                    self.x_offline.dtype
                )
                for c in completions
            ]

        valid_indices = np.array([i for i, r in enumerate(results) if r is not None])
        y = np.full((len(completions), 1), np.nan)

        if len(valid_indices) > 0:
            x = np.stack([r for r in results if r is not None])
            y[valid_indices] = self._evaluate_designs(x)

        return y, len(valid_indices)

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...

    def _render_design(self, x_i: np.ndarray) -> str:
        if self.categories is not None:
            # Preserve the quotes around each category for categorical values
            return f"<design>{[self.categories[i] for i in x_i]}</design>"

        # Use the shortest round-trip representation for numerical values
        return f"<design>[{', '.join(str(p) for p in x_i)}]</design>"

    def _cached_parallel_predict(self, x: np.ndarray, cache_path: Path) -> np.ndarray:
        if cache_path.exists():
            logger.info("Loading predictions from cache: %s", cache_path)
            y = np.load(cache_path)

            if y.shape != (len(x), 1):
                raise ValueError(
                    f"Predictions loaded from {cache_path} must have shape "
                    f"({len(x)}, 1), got {y.shape}"
                )

            return y

        with ProcessPoolExecutor(
            max_workers=_NUM_PREDICT_WORKERS,
            mp_context=get_context("fork"),
            initializer=_init_worker_predict,
            initargs=(self.predict,)
        ) as executor:
            prediction_iter = tqdm(
                executor.map(_predict_one, x),
                desc="Predicting",
                total=len(x)
            )
            y = np.concatenate(list(prediction_iter))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, y)
        logger.info("Saved predictions to cache: %s", cache_path)

        return y

    def _evaluate_designs(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)


_REGISTRY: dict[str, type[BenchmarkTask]] = {}


def make_task(task_key: str, num_designs: int) -> BenchmarkTask:
    return _REGISTRY[task_key](task_key, num_designs)


def register_tasks(
    *task_keys: str
) -> Callable[[type[BenchmarkTask]], type[BenchmarkTask]]:
    def decorator(cls: type[BenchmarkTask]) -> type[BenchmarkTask]:
        for task_key in task_keys:
            if task_key in _REGISTRY:
                raise ValueError(f"Task already registered: {task_key}")

            _REGISTRY[task_key] = cls

        return cls

    return decorator


_worker_predict: Callable[[np.ndarray], np.ndarray] | None = None


def _init_worker_predict(predict: Callable[[np.ndarray], np.ndarray]) -> None:
    global _worker_predict
    _worker_predict = predict


def _predict_one(x_i: np.ndarray) -> np.ndarray:
    if _worker_predict is None:
        raise RuntimeError("_worker_predict is not initialized")

    return _worker_predict(x_i.reshape(1, -1))
