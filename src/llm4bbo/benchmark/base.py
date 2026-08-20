from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import resources

import numpy as np
from transformers.pipelines.text_generation import ChatType


DATA_DIR = resources.files("llm4bbo") / "assets" / "data"


class BenchmarkTask(ABC):
    def __init__(
        self,
        task_name: str,
        design_dim: int,
        num_designs: int,
        x_offline: np.ndarray,
        y_offline: np.ndarray,
        system_prompt: str,
        user_prompt: str
    ) -> None:
        if x_offline.shape != (len(x_offline), design_dim):
            raise ValueError(
                "x_offline must have shape "
                f"(n, {design_dim}), got {x_offline.shape}"
            )

        if y_offline.shape != (len(y_offline), 1):
            raise ValueError(f"y_offline must have shape (n, 1), got {y_offline.shape}")

        if len(x_offline) != len(y_offline):
            raise ValueError("x_offline and y_offline must have equal lengths")

        if not 0 < num_designs <= len(x_offline):
            raise ValueError(f"num_designs must be in [1, {len(x_offline)}]")

        self.task_name = task_name
        self.design_dim = design_dim
        self.num_designs = num_designs

        self.x_offline = x_offline
        self.y_offline = y_offline

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        self.selected_indices = self._select_evenly_spaced_indices()
        self.x = self.x_offline[self.selected_indices]
        self.y = self.y_offline[self.selected_indices]

    def _select_evenly_spaced_indices(self) -> np.ndarray:
        sorted_index = self.y_offline.squeeze(-1).argsort()
        spaced_index = (
            np.linspace(0, len(self.y_offline) - 1, self.num_designs)
            .round()
            .astype(int)
        )
        return sorted_index[spaced_index]

    def create_prompt_messages(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        use_tool: bool,
        max_tool_calls: int | None = None,
        generate_trace: bool = False,
        x_target: np.ndarray | None = None
    ) -> ChatType:
        system_parts = [self.system_prompt]

        if use_tool:
            if max_tool_calls is None or max_tool_calls <= 0:
                raise ValueError(
                    "max_tool_calls must be a positive integer when use_tool=True"
                )

            system_parts.append(TOOL_USE_PROMPT.format(max_tool_calls=max_tool_calls))

        if generate_trace:
            if x_target is None:
                raise ValueError("x_target is required when generate_trace=True")

            target = self.render_design(x_target)
            system_parts.append(TRACE_GENERATION_PROMPT.format(target=target))

        system_parts.append(FINAL_INSTRUCTION)

        references = "\n".join(
            self.render_example(x, y)
            for x, y in zip(xs, ys, strict=True)
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

        results = [self.parse_completion(c) for c in completions]
        designs, valids = zip(*results, strict=True)
        return np.array(designs), np.array(valids)

    @abstractmethod
    def render_design(self, x: np.ndarray) -> str:
        ...

    @abstractmethod
    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        ...

    @abstractmethod
    def parse_completion(self, completion: str) -> tuple[list[int] | list[float], bool]:
        ...

    @abstractmethod
    def predict(self, xs: np.ndarray) -> np.ndarray:
        ...


REGISTRY: dict[str, type[BenchmarkTask]] = {}


def make_task(task_name: str, num_designs: int) -> BenchmarkTask:
    return REGISTRY[task_name](task_name, num_designs)


def register_tasks(*task_names: str) -> Callable:
    def decorator(cls: type[BenchmarkTask]) -> type[BenchmarkTask]:
        for name in task_names:
            if name in REGISTRY:
                raise ValueError(f"Duplicate task registration: {name}")

            REGISTRY[name] = cls

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
