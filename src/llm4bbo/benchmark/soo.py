__all__ = ["SOOBenchTask"]

from soo_bench.Taskdata import OfflineTask
from soo_bench.Taskunit import TaskUnit

import numpy as np

from .base import BenchmarkTask, register_tasks
from .prompts.soo import SOO_SYSTEM_PROMPTS


# GTOPX5 and CEC1 are excluded because sampling takes too long
_TASK_METADATA = {
    "gtopx1": ("gtopx_data", 1),
    "gtopx2": ("gtopx_data", 2),
    "gtopx3": ("gtopx_data", 3),
    "gtopx4": ("gtopx_data", 4),
    # "gtopx5": ("gtopx_data", 5),
    "gtopx6": ("gtopx_data", 6),
    "gtopx7": ("gtopx_data", 7),
    # "cec1": ("cec_data", 1),
    "cec2": ("cec_data", 2),
    "cec3": ("cec_data", 3),
    "cec4": ("cec_data", 4),
    "cec5": ("cec_data", 5)
}


@register_tasks(*_TASK_METADATA)
class SOOBenchTask(BenchmarkTask):
    benchmark: str = "soo_bench"

    def __init__(self, task_key: str, num_designs: int) -> None:
        self._task = OfflineTask(*_TASK_METADATA[task_key])

        self.x_low = np.array(self._task.xl)
        self.x_high = np.array(self._task.xu)

        taskunit = self._task.taskunit

        if taskunit.task == "gtopx_data" and taskunit.benchmark == 7:
            # GTOPX 7: Planet code 9 causes out-of-bounds access in the C++ oracle
            self.x_high[6:10] = np.nextafter(8.5, -np.inf)

        # Patch taskunit methods to customize offline dataset sampling
        # sample_x calls filter_useful only for tasks with constraints
        # GTOPX 2, 3, 4, 6 are unconstrained, so sample_x skips filter_useful
        taskunit.filter_useful = self._patched_filter_useful

        if taskunit.task == "cec_data" and taskunit.benchmark == 1:
            # Task_CEC.sample_x forces CEC1's feasible fraction to zero
            # Bypass it by patching in the base TaskUnit.sample_x implementation
            # https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L559-L560
            taskunit.sample_x = TaskUnit.sample_x.__get__(taskunit)

        # sample_bound calls sample_x through sample_x_using_cache
        # sample_x uses filter_useful to separate valid and invalid designs
        # and samples until both requested counts are reached
        # Sample dim * 1000 valid designs and keep those in the lower 50th percentile
        self._task.sample_bound(high=50, rate_satisfying_constraints=1)

        super().__init__(
            task_name=f"{taskunit.task}-{taskunit.benchmark}",
            num_designs=num_designs,
            system_prompt=SOO_SYSTEM_PROMPTS[taskunit.task, taskunit.benchmark],
            x_offline=self._task.x,
            x_low=self.x_low,
            x_high=self.x_high
        )

    def check_satisfy_constraints(self, x: np.ndarray) -> np.ndarray:
        if not self._task.is_constraint():
            return np.ones(len(x), dtype=bool)

        _, constraints = self._predict(x)
        return np.all(constraints >= 0, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._predict(x)[0].reshape(-1, 1)

    # https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L69-L82
    def _patched_filter_useful(
        self,
        x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        taskunit = self._task.taskunit

        match taskunit.task, taskunit.benchmark:
            case "gtopx_data", 7:
                # GTOPX 7: Round planet code parameters to the nearest integer
                x[:, 6:10] = np.floor(x[:, 6:10] + 0.5)
            case "cec_data", 2:
                # CEC 2: Round the binary parameter to the nearest integer
                x[:, 2] = np.round(x[:, 2])
            case "cec_data", 3:
                # CEC 3: Round the binary parameter to the nearest integer
                x[:, 1] = np.round(x[:, 1])

        # y_offline is not set yet, so non-finite predictions are preserved
        y, constraints = self._predict(x)
        is_valid = np.isfinite(y) & np.all(constraints >= 0, axis=1)
        return list(x[is_valid]), list(x[~is_valid])

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Predict only the designs whose parameters are within bounds
        # to prevent out-of-bounds inputs from receiving valid scores
        # in tasks with con_num == 0 (GTOPX 2, 3, 4, 6)
        y = np.full(len(x), -np.inf)
        constraints = np.full((len(x), self._task.con_num), -np.inf)

        x_is_valid = self.check_within_bounds(x)

        if x_is_valid.any():
            taskunit = self._task.taskunit

            # GTOPX: self._task.predict replaces predicted NaNs with max(taskunit.y)
            # Never let a failed prediction become a finite, high score
            # https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L414-L415
            y_original = taskunit.y
            taskunit.y = np.array(-np.inf)
            y_valid, constraints_valid = self._task.predict(x[x_is_valid])
            taskunit.y = y_original

            y[x_is_valid] = y_valid

            if self._task.is_constraint():
                constraints[x_is_valid] = constraints_valid

        y_is_valid = np.isfinite(y)
        constraints_is_valid = np.all(constraints >= 0, axis=1)
        is_valid = x_is_valid & y_is_valid & constraints_is_valid

        if hasattr(self, "y_offline"):
            y[~is_valid] = self.y_offline.min()

        return y, constraints
