__all__ = ["DesignBenchTask"]

import design_bench

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

        if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
            categories = ["A", "C", "G", "T"]

        super().__init__(
            task_name=task_name,
            num_designs=num_designs,
            system_prompt=DESIGN_SYSTEM_PROMPTS[task_name],
            x_offline=x_offline,
            categories=categories
        )

        if self.task_name != "TFBind10-Exact-v0":
            cache_path = self.data_dir / f"{self.task_name}_y.npy"
            y_all = self._cached_parallel_predict(x_all, cache_path)

        # Normalize predicted scores using the range of all targets
        self._oracle_scaler = MinMaxScaler().fit(y_all)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.task_name == "TFBind10-Exact-v0":
            if x.ndim != 2:
                raise ValueError(f"x must be a 2D array, got shape {x.shape}")

            return self._tfbind10_y[x @ self._tfbind10_weights]

        return self._task.predict(x)

    def _evaluate_designs(self, x: np.ndarray) -> np.ndarray:
        return self._oracle_scaler.transform(self.predict(x))
