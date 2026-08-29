import ast
import re
from pathlib import Path

from soo_bench.Taskdata import OfflineTask

import numpy as np

from .base import BenchmarkTask, register_tasks
from .prompts.soo import SOO_SYSTEM_PROMPTS
from .prompts.user import USER_TEMPLATE


# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/paper/SOO_Bench.pdf
TASK_SPECS = {
    "gtopx1": ("GTOPX1", "gtopx_data", 1),
    "gtopx2": ("GTOPX2", "gtopx_data", 2),
    "gtopx3": ("GTOPX3", "gtopx_data", 3),
    "gtopx4": ("GTOPX4", "gtopx_data", 4),
    "gtopx5": ("GTOPX5", "gtopx_data", 5),
    "gtopx6": ("GTOPX6", "gtopx_data", 6),
    "gtopx7": ("GTOPX7", "gtopx_data", 7),
    "cec1": ("CEC1", "cec_data", 1),
    "cec2": ("CEC2", "cec_data", 2),
    "cec3": ("CEC3", "cec_data", 3),
    "cec4": ("CEC4", "cec_data", 4),
    "cec5": ("CEC5", "cec_data", 5)
}

PARAMETER_PRECISION = 6
SCORE_PRECISION = 6
DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


@register_tasks(*TASK_SPECS)
class SOOTask(BenchmarkTask):
    benchmark: str = "soo"

    def __init__(self, task_key: str, num_designs: int) -> None:
        self.task_name, task_family, benchmark_id = TASK_SPECS[task_key]
        self.num_designs = num_designs

        self._task = OfflineTask(task_family, benchmark_id, use_cache=False)
        self.design_dim = self._task.var_num
        self._num_constraints = max(
            self._task.con_num - 2 * self.design_dim,
            0,
        )

        if task_family == "gtopx_data":
            # GTOPX replaces NaNs with max(self.y) before y is initialized.
            # Use -inf so our invalid-score handling can apply its own penalty.
            self._task.taskunit.y = np.array([-np.inf])

        self.system_prompt, self.user_prompt = _prepare_prompts(self.task_name)

        x_offline = _prepare_designs(self.task_name, self.data_dir)
        self._prepare_score_cache(x_offline)

        super().__init__(x_offline)

    def render_design(self, x: np.ndarray) -> str:
        return _render_soo_design(x)

    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        return _render_soo_example(x, y)

    def _parse_completion(self, completion: str) -> tuple[list[float], bool]:
        return _parse_soo_completion(completion, self.design_dim)

    # https://github.com/zhuyiyi-123/SOO-Bench/blob/main/scripts/result_gather.py#L296-L320
    def _predict(self, x: np.ndarray) -> np.ndarray:
        scores, constraints = self._raw_predict(x)
        invalid = ~np.isfinite(scores)

        if self._num_constraints > 0:
            invalid |= np.any(constraints < 0, axis=1)

        scores[invalid] = self._min_score
        return scores.reshape(-1, 1)

    def _raw_predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._task.predict(np.array(x, copy=True))

    def _prepare_score_cache(self, x_offline: np.ndarray) -> None:
        cache_path = self.data_dir / f"{self.task_name}_y_offline.npy"

        if cache_path.exists():
            self._min_score = np.load(cache_path).min().item()
            return

        scores, constraints = self._raw_predict(x_offline)
        self._min_score = np.nanmin(scores).item()
        invalid = ~np.isfinite(scores)

        if self._num_constraints > 0:
            invalid |= np.any(constraints < 0, axis=1)

        scores[invalid] = self._min_score
        np.save(cache_path, scores.reshape(-1, 1))


def _prepare_prompts(task_name: str) -> tuple[str, str]:
    return SOO_SYSTEM_PROMPTS[task_name], USER_TEMPLATE


def _prepare_designs(task_name: str, data_dir: Path) -> np.ndarray:
    return np.load(data_dir / f"{task_name}_x.npy")


def _render_soo_design(x: np.ndarray) -> str:
    parameters = [round(float(p), PARAMETER_PRECISION) for p in x]
    return f"<design>{parameters}</design>"


def _render_soo_example(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"Design: {_render_soo_design(x)}, "
        f"Score: {round(y.item(), SCORE_PRECISION)}"
    )


def _parse_soo_completion(
    completion: str,
    design_dim: int
) -> tuple[list[float], bool]:
    matches = DESIGN_PATTERN.findall(completion)

    if not matches:
        return [0.0] * design_dim, False

    try:
        design = ast.literal_eval(matches[-1])
    except Exception:
        return [0.0] * design_dim, False

    if not isinstance(design, list) or len(design) != design_dim:
        return [0.0] * design_dim, False

    try:
        design = [float(parameter) for parameter in design]
    except Exception:
        return [0.0] * design_dim, False

    if not all(np.isfinite(design)):
        return [0.0] * design_dim, False

    return design, True
