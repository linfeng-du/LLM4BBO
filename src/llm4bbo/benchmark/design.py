__all__ = ["DesignBenchTask"]

import design_bench
from morphing_agents.mujoco.ant import elements as ant_elements
from morphing_agents.mujoco.dkitty import elements as dkitty_elements

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import BenchmarkTask, register_tasks
from .prompts.design import DESIGN_SYSTEM_PROMPTS


_TASK_METADATA = {
    "tf8": "TFBind8-Exact-v0",
    "tf10": "TFBind10-Exact-v0",
    "ant": "AntMorphology-Exact-v0",
    "dkitty": "DKittyMorphology-Exact-v0"
}


@register_tasks(*_TASK_METADATA)
class DesignBenchTask(BenchmarkTask):
    benchmark: str = "design_bench"

    def __init__(self, task_key: str, num_designs: int) -> None:
        task_name = _TASK_METADATA[task_key]
        self._task = design_bench.make(task_name)

        if task_name == "TFBind10-Exact-v0":
            x_all = np.load(self.data_dir / f"{task_name}_x.npy")
            y_all = np.load(self.data_dir / f"{task_name}_y.npy")

            # Use designs in the lower 50th percentile as the offline dataset
            size = len(y_all) // 2
            indices = y_all.squeeze(-1).argpartition(size)[:size]
            x_offline = x_all[indices]

            # Store scores and base-4 positional weights for prediction
            self._tfbind10_y = y_all
            self._tfbind10_weights = 4 ** np.arange(
                start=x_all.shape[1] - 1,
                stop=-1,
                step=-1
            )

        else:
            x_offline = self._task.x

            # Create a temporary task object to avoid mutating self._task
            tmp_task = design_bench.make(task_name)
            tmp_task.dataset.subsample()
            x_all = tmp_task.x
            del tmp_task

        categories = None

        match task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                categories = ["A", "C", "G", "T"]
                x_low = np.zeros(x_offline.shape[1])
                x_high = np.full(x_offline.shape[1], len(categories) - 1)
            case "AntMorphology-Exact-v0":
                x_low = np.tile(ant_elements.LEG_LOWER_BOUND, 4)
                x_high = np.tile(ant_elements.LEG_UPPER_BOUND, 4)
            case "DKittyMorphology-Exact-v0":
                x_low = np.tile(dkitty_elements.LEG_LOWER_BOUND, 4)
                x_high = np.tile(dkitty_elements.LEG_UPPER_BOUND, 4)

        x_low = x_low.astype(x_offline.dtype)
        x_high = x_high.astype(x_offline.dtype)

        super().__init__(
            task_name=task_name,
            num_designs=num_designs,
            system_prompt=DESIGN_SYSTEM_PROMPTS[task_name],
            x_offline=x_offline,
            x_low=x_low,
            x_high=x_high,
            categories=categories
        )

        if self.task_name != "TFBind10-Exact-v0":
            cache_path = self.data_dir / f"{self.task_name}_y.npy"
            y_all = self._cached_parallel_predict(x_all, cache_path)

        # Normalize predicted scores using the range of all targets
        self._oracle_scaler = MinMaxScaler().fit(y_all)

    def evaluate(self, completions: list[str]) -> tuple[np.ndarray, int]:
        y, num_valid = super().evaluate(completions)
        return self._oracle_scaler.transform(y), num_valid

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.task_name == "TFBind10-Exact-v0":
            if x.ndim != 2:
                raise ValueError(f"x must be a 2D array, got shape {x.shape}")

            return self._tfbind10_y[x @ self._tfbind10_weights]

        return self._task.predict(x)
