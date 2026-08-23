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


logger = logging.getLogger(__name__)

ASSETS_DIR = resources.files("llm4bbo") / "assets"
NUM_PREDICT_WORKERS = len(os.sched_getaffinity(0))


class BenchmarkTask(ABC):
    benchmark: str
    task_name: str
    design_dim: int
    num_designs: int
    system_prompt: str
    user_prompt: str

    def __init__(self, x_offline: np.ndarray) -> None:
        if x_offline.shape != (len(x_offline), self.design_dim):
            raise ValueError(
                "x_offline must have shape "
                f"({len(x_offline)}, {self.design_dim}), got {x_offline.shape}"
            )

        if not 0 < self.num_designs <= len(x_offline):
            raise ValueError(f"num_designs must be in [1, {len(x_offline)}]")

        self.x_offline = x_offline
        self.y_offline = self.predict(
            self.x_offline,
            cache_path=self.data_dir / f"{self.task_name}_y_offline.npy"
        )

        self.selected_indices = _select_evenly_spaced_indices(
            self.y_offline,
            self.num_designs
        )
        self.x = self.x_offline[self.selected_indices]
        self.y = self.y_offline[self.selected_indices]

    @property
    def benchmark_dir(self) -> Path:
        return ASSETS_DIR / self.benchmark

    @property
    def data_dir(self) -> Path:
        return self.benchmark_dir / "data"

    def create_prompt_messages(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_tool_calls: int | None = None,
        x_target: np.ndarray | None = None
    ) -> ChatType:
        system_parts = [self.system_prompt]

        if max_tool_calls is not None:
            if max_tool_calls <= 0:
                raise ValueError("max_tool_calls must be positive")

            system_parts.append(TOOL_USE_PROMPT.format(max_tool_calls=max_tool_calls))

        if x_target is not None:
            target = self.render_design(x_target)
            system_parts.append(TRACE_GENERATION_PROMPT.format(target=target))

        system_parts.append(FINAL_INSTRUCTION)

        references = "\n".join(
            self.render_example(x_, y_)
            for x_, y_ in zip(x, y, strict=True)
        )

        system_prompt = "\n\n".join(system_parts)
        user_prompt = self.user_prompt.format(references=references)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def create_completion_messages(self, x: np.ndarray) -> ChatType:
        return [{"role": "assistant", "content": self.render_design(x)}]

    def parse_completions(
        self,
        completions: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not completions:
            raise ValueError("completions must not be empty")

        results = [self._parse_completion(c) for c in completions]
        designs, valid_flags = zip(*results, strict=True)
        return np.array(designs), np.array(valid_flags)

    def predict(self, x: np.ndarray, cache_path: Path | None = None) -> np.ndarray:
        if cache_path is None:
            return self._predict(x)

        if cache_path.exists():
            logger.info("Loading predictions from cache: %s", cache_path)
            y = np.load(cache_path)

            if y.shape != (len(x), 1):
                raise ValueError(
                    f"Cache file {cache_path} must have shape "
                    f"({len(x)}, 1), got {y.shape}"
                )

            return y

        with ProcessPoolExecutor(
            max_workers=NUM_PREDICT_WORKERS,
            mp_context=get_context("fork"),
            initializer=_init_worker_predict,
            initargs=(self._predict,)
        ) as executor:
            prediction_iter = tqdm(
                executor.map(_predict_one, x),
                total=len(x),
                desc="Predicting"
            )
            y = np.concatenate(list(prediction_iter))

            np.save(cache_path, y)
            logger.info("Saved predictions to cache: %s", cache_path)

        return y

    @abstractmethod
    def render_design(self, x: np.ndarray) -> str:
        ...

    @abstractmethod
    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        ...

    @abstractmethod
    def _parse_completion(
        self,
        completion: str
    ) -> tuple[list[int] | list[float], bool]:
        ...

    @abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        ...


_worker_predict = None


def _init_worker_predict(predict: Callable) -> None:
    global _worker_predict
    _worker_predict = predict


def _predict_one(x: np.ndarray) -> np.ndarray:
    return _worker_predict(x[None, :])


def _select_evenly_spaced_indices(y: np.ndarray, num_designs: int) -> np.ndarray:
    sorted_indices = y.squeeze(-1).argsort()
    spaced_indices = np.linspace(0, len(y) - 1, num_designs).round().astype(int)
    return sorted_indices[spaced_indices]


REGISTRY: dict[str, type[BenchmarkTask]] = {}


def make_task(task_key: str, num_designs: int) -> BenchmarkTask:
    return REGISTRY[task_key](task_key, num_designs)


def register_tasks(*task_keys: str) -> Callable:
    def decorator(cls: type[BenchmarkTask]) -> type[BenchmarkTask]:
        for task_key in task_keys:
            if task_key in REGISTRY:
                raise ValueError(f"Task already registered: {task_key}")

            REGISTRY[task_key] = cls

        return cls

    return decorator


TOOL_USE_PROMPT = """\
You may use the `predict_score` tool \
to evaluate up to {max_tool_calls} intermediate designs. \
The tool returns the predicted score and uncertainty for each design. \
Use these results to continue your reasoning.\
"""

TRACE_GENERATION_PROMPT = """\
Generate a reasoning trace that arrives at the target design below. \
The target design is known to outperform all provided examples.

Target design: {target}

Treat the target design as the outcome of your own analysis. \
Do not mention or imply that it was supplied in advance. \
Conclude with the target design as your final answer.\
"""

FINAL_INSTRUCTION = """\
Reason step-by-step, \
but keep your reasoning concise, \
preferably within 500 tokens. \
Enclose the final design in <design></design> XML tags.\
"""
