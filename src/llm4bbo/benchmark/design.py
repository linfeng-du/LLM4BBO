__all__ = ["DesignBenchTask"]

import design_bench

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import BenchmarkTask, register_tasks
from .prompts.design import DESIGN_SYSTEM_PROMPTS
from .utils import parse_categorical, parse_numerical


_TASK_METADATA = {
    "tf8": ("TFBind8-Exact-v0", 8),
    "tf10": ("TFBind10-Exact-v0", 10),
    "ant": ("AntMorphology-Exact-v0", 60),
    "dkitty": ("DKittyMorphology-Exact-v0", 56)
}

# TODO: Confirm the decimal precision for rendered parameters and scores
_PARAMETER_PRECISION = 3
_SCORE_PRECISION = 6

_BASES = ["A", "C", "G", "T"]


@register_tasks(*_TASK_METADATA)
class DesignBenchTask(BenchmarkTask):
    benchmark: str = "design_bench"

    def __init__(self, task_key: str, num_designs: int) -> None:
        self.task_name, self.design_dim = _TASK_METADATA[task_key]
        self.num_designs = num_designs
        self.system_prompt = DESIGN_SYSTEM_PROMPTS[self.task_name]

        self._task = design_bench.make(self.task_name)

        if self.task_name == "TFBind10-Exact-v0":
            x_all = np.load(self.data_dir / f"{self.task_name}_x.npy")
            y_all = np.load(self.data_dir / f"{self.task_name}_y.npy")

            # Use designs in the lower 50th percentile as the offline dataset
            size = len(y_all) // 2
            indices = y_all.squeeze(-1).argpartition(size)[:size]
            x_offline = x_all[indices]

            # Store scores and base-4 positional weights for prediction
            self._tfbind10_scores = y_all
            self._tfbind10_weights = 4 ** np.arange(self.design_dim - 1, -1, -1)

        else:
            x_offline = self._task.x

            # Create a temporary task object to avoid mutating self._task
            tmp_task = design_bench.make(self.task_name)
            tmp_task.dataset.subsample()
            x_all = tmp_task.x
            del tmp_task

            cache_path = self.data_dir / f"{self.task_name}_y.npy"
            y_all = self.predict(x_all, cache_path=cache_path)

        self.oracle_scaler = MinMaxScaler().fit(y_all)
        super().__init__(x_offline)

    def render_design(self, x: np.ndarray) -> str:
        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                design = [_BASES[b] for b in x]
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                design = [round(p.item(), _PARAMETER_PRECISION) for p in x]
            case _:
                raise ValueError(f"Invalid task: {self.task_name}")

        return f"<design>{design}</design>"

    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        design = self.render_design(x)
        score = round(y.item(), _SCORE_PRECISION)

        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                return f"{design}, Binding Score: {score}"
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                return f"{design}, Performance Score: {score}"

        raise ValueError(f"Invalid task: {self.task_name}")

    def _parse_completion(
        self,
        completion: str
    ) -> tuple[list[int] | list[float], bool]:
        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                return parse_categorical(completion, self.design_dim, categories=_BASES)
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                return parse_numerical(completion, self.design_dim, dtype=float)

        raise ValueError(f"Invalid task: {self.task_name}")

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self.task_name == "TFBind10-Exact-v0":
            return self._tfbind10_scores[x @ self._tfbind10_weights]

        return self._task.predict(x)
