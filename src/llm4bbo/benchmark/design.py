import ast
import re
from pathlib import Path

import design_bench
from design_bench.task import Task

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import BenchmarkTask, register_tasks


TASK_SPECS = {
    "tf8": ("TFBind8-Exact-v0", 8),
    "tf10": ("TFBind10-Exact-v0", 10),
    "ant": ("AntMorphology-Exact-v0", 60),
    "dkitty": ("DKittyMorphology-Exact-v0", 56)
}

# TODO: Confirm the decimal precision for rendered parameters and scores
PARAMETER_PRECISION = 3
SCORE_PRECISION = 6

BASES = ["A", "C", "G", "T"]
DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


@register_tasks(*TASK_SPECS)
class DesignBenchTask(BenchmarkTask):
    benchmark: str = "design_bench"

    def __init__(self, task_key: str, num_designs: int) -> None:
        self.task_name, self.design_dim = TASK_SPECS[task_key]
        self.num_designs = num_designs

        self.system_prompt, self.user_prompt = _prepare_prompts(
            self.task_name,
            self.design_dim
        )

        self._task, x_offline, x_all = _prepare_task_and_designs(
            self.task_name,
            self.data_dir
        )

        if self.task_name == "TFBind10-Exact-v0":
            self._tfbind10_oracle = _load_tfbind10_oracle(self.data_dir)

        super().__init__(x_offline)

        y_all = self.predict(
            x_all,
            cache_path=self.data_dir / f"{self.task_name}_y.npy"
        )
        self.oracle_scaler = MinMaxScaler().fit(y_all)

    def render_design(self, x: np.ndarray) -> str:
        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                return _render_tfbind_design(x)
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                return _render_morphology_design(x)
            case _:
                raise ValueError(f"Invalid task: {self.task_name}")

    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                return _render_tfbind_example(x, y)
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                return _render_morphology_example(x, y)
            case _:
                raise ValueError(f"Invalid task: {self.task_name}")

    def _parse_completion(
        self,
        completion: str
    ) -> tuple[list[int] | list[float], bool]:
        match self.task_name:
            case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
                return _parse_tfbind_completion(completion, self.design_dim)
            case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
                return _parse_morphology_completion(completion, self.design_dim)
            case _:
                raise ValueError(f"Invalid task: {self.task_name}")

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self.task_name == "TFBind10-Exact-v0":
            return _predict_tfbind10(self._tfbind10_oracle, x)

        return self._task.predict(x)


def _prepare_prompts(task_name: str, design_dim: int) -> tuple[str, str]:
    match task_name:
        case "TFBind8-Exact-v0" | "TFBind10-Exact-v0":
            factor = "SIX6_REF_R1" if task_name == "TFBind8-Exact-v0" else "Pho4"
            system_prompt = TFBIND_SYSTEM_PROMPT.format(
                design_dim=design_dim,
                factor=factor
            )
            return system_prompt, TFBIND_USER_PROMPT

        case "AntMorphology-Exact-v0" | "DKittyMorphology-Exact-v0":
            system_prompt = (
                ANT_SYSTEM_PROMPT
                if task_name == "AntMorphology-Exact-v0"
                else DKITTY_SYSTEM_PROMPT
            )
            return system_prompt, MORPHOLOGY_USER_PROMPT

        case _:
            raise ValueError(f"Invalid task: {task_name}")


def _prepare_task_and_designs(
    task_name: str,
    data_dir: Path
) -> tuple[Task, np.ndarray, np.ndarray]:
    task = design_bench.make(task_name)

    if task_name == "TFBind10-Exact-v0":
        x_all = np.load(data_dir / f"{task_name}_x.npy")
        tfbind10_oracle = _load_tfbind10_oracle(data_dir)
        y_all = _predict_tfbind10(tfbind10_oracle, x_all)

        # Use designs in the lower 50th percentile as the offline dataset
        offline_size = len(y_all) // 2
        offline_indices = y_all.squeeze(-1).argpartition(offline_size)[:offline_size]
        x_offline = x_all[offline_indices]

        return task, x_offline, x_all

    x_offline = task.x

    # Create a temporary task object to avoid mutating `task`
    tmp_task = design_bench.make(task_name)
    tmp_task.dataset.subsample()
    x_all = tmp_task.x

    return task, x_offline, x_all


def _load_tfbind10_oracle(data_dir: Path) -> dict[str, float]:
    text = (data_dir / "TFBind10-Exact-v0_oracle.txt").read_text()
    return {k: float(v) for l in text.splitlines() for k, v in [l.split()]}


def _render_tfbind_design(x: np.ndarray) -> str:
    return f"<design>{[BASES[b] for b in x]}</design>"


def _render_morphology_design(x: np.ndarray) -> str:
    return f"<design>{[round(p.item(), PARAMETER_PRECISION) for p in x]}</design>"


def _render_tfbind_example(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"DNA: {_render_tfbind_design(x)}, "
        f"Binding Score: {round(y.item(), SCORE_PRECISION)}"
    )


def _render_morphology_example(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"Robot Morphology: {_render_morphology_design(x)}, "
        f"Performance Score: {round(y.item(), SCORE_PRECISION)}"
    )


def _parse_tfbind_completion(
    completion: str,
    design_dim: int
) -> tuple[list[int], bool]:
    matches = DESIGN_PATTERN.findall(completion)

    if not matches:
        return [0] * design_dim, False

    try:
        design = ast.literal_eval(matches[-1])
    except Exception:
        # Handle cases like <design>ACGT</design>
        design = list(matches[-1].strip())

    if isinstance(design, str):
        # Handle cases like <design>'ACGT'</design>
        design = list(design)

    if (
        not isinstance(design, list)
        or len(design) != design_dim
        or not all(b in BASES for b in design)
    ):
        return [0] * design_dim, False

    return [BASES.index(b) for b in design], True


def _parse_morphology_completion(
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
        design = [float(p) for p in design]
    except Exception:
        return [0.0] * design_dim, False

    if not all(np.isfinite(design)):
        return [0.0] * design_dim, False

    return design, True


def _predict_tfbind10(oracle: dict[str, float], x: np.ndarray) -> np.ndarray:
    x_char = np.array(BASES)[x]
    return np.array([[oracle["".join(xc)]] for xc in x_char])


# TFBind8-Exact-v0 & TFBind10-Exact-v0
TFBIND_SYSTEM_PROMPT = """\
You are an expert in DNA sequence design. \
Your task is to design a DNA sequence \
of exactly {design_dim} bases using only A, C, G, and T. \
Your objective is to maximize its binding score for the transcription factor {factor}.\
"""

TFBIND_USER_PROMPT = """\
The following DNA sequences are provided as references, \
along with their binding scores:

{references}

Using these examples as references, \
design a new sequence expected to outperform the best example.\
"""

# AntMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/ant
ANT_SYSTEM_PROMPT = """\
You are an expert in quadruped robot morphology design. \
Your task is to design a morphology for the Ant quadruped robot \
that maximizes its running speed. \
The morphology is represented by 60 continuous parameters, \
arranged as four consecutive 15-parameter blocks, one for each leg. \
Each leg is a 3-link kinematic chain with hip, thigh, and ankle joints. \
Round every parameter to three decimal places.

Within each 15-parameter leg block, the parameters follow this schema:
p0, p1, p2: 3D location on the torso where the leg is mounted.
p3, p4, p5: Fixed orientation of the leg relative to the torso.
p6, p7: Midpoint and half-range of the hip joint's motion range.
p8, p9: Midpoint and half-range of the thigh joint's motion range.
p10, p11: Midpoint and half-range of the ankle joint's motion range.
p12, p13, p14: Lengths of the hip, thigh, and ankle links.\
"""

# DKittyMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/dkitty
DKITTY_SYSTEM_PROMPT = """\
You are an expert in quadruped robot morphology design. \
Your task is to design a morphology for the D'Kitty quadruped robot \
that maximizes its navigation performance toward a fixed target location. \
The morphology is represented by 56 continuous parameters, \
arranged as four consecutive 14-parameter blocks, one for each leg. \
Each leg is a 3-link kinematic chain with hip, thigh, and ankle joints. \
Round every parameter to three decimal places.

Within each 14-parameter leg block, the parameters follow this schema:
p0, p1, p2: 3D location on the torso where the leg is mounted.
p3, p4, p5: Fixed orientation of the leg relative to the torso.
p6, p7: Midpoint and half-range of the hip joint's motion range.
p8, p9: Midpoint and half-range of the thigh joint's motion range.
p10, p11: Midpoint and half-range of the ankle joint's motion range.
p12, p13: Lengths of the thigh and ankle links.\
"""

MORPHOLOGY_USER_PROMPT = """\
The following robot morphologies are provided as references, \
along with their performance scores:

{references}

Using these examples as references, \
design a new morphology expected to outperform the best example.\
"""
