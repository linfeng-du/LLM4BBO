import ast
import re
from pathlib import Path

import RNA

import numpy as np

from .base import BenchmarkTask, register_tasks


TASK_SPECS = {
    "rna1": ("L14_RNA1", 14),
    "rna2": ("L14_RNA2", 14),
    "rna3": ("L14_RNA3", 14)
}

# https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L137-L143
TARGETS = {
    "L14_RNA1": (
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACC"
        "CCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC"
    ),
    "L14_RNA2": (
        "GAGGCACAUUCCGGCUCGCCCCCGUCCGCGCGGGGGCCCCGCGCGGACGG"
        "GGUCCGGCCCGCGCGGGGCCCCCGCGCGGGAGCCGGAAUGUGCCUCGUUC"
    ),
    "L14_RNA3": (
        "CCGGUGAUACUGUUAGUGGUCACGGUGCAUUUAUAGCGCUAAAGUACAGU"
        "CUUCCCCUGUUGAACGGCGCCAUUGCAUACAGGGCCAGCCGCGUAACGCC"
    )
}

# TODO: Confirm the decimal precision for rendered scores
SCORE_PRECISION = 6

BASES = ["U", "G", "C", "A"]
DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


@register_tasks(*TASK_SPECS)
class RNATask(BenchmarkTask):
    benchmark: str = "rna"

    def __init__(self, task_key: str, num_designs: int) -> None:
        self.task_name, self.design_dim = TASK_SPECS[task_key]
        self.num_designs = num_designs

        self._target = TARGETS[self.task_name]
        self._norm_value = _compute_min_binding_energy(self._target, self.design_dim)

        self.system_prompt, self.user_prompt = _prepare_prompts(
            self._target,
            self.design_dim
        )

        x_offline = _prepare_designs(self.task_name, self.data_dir)

        super().__init__(x_offline)

    def render_design(self, x: np.ndarray) -> str:
        return _render_rna_design(x)

    def render_example(self, x: np.ndarray, y: np.ndarray) -> str:
        return _render_rna_example(x, y)

    def _parse_completion(self, completion: str) -> tuple[list[int], bool]:
        return _parse_rna_completion(completion, self.design_dim)

    # https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L87-L116
    def _predict(self, x: np.ndarray) -> np.ndarray:
        sequences = ["".join(BASES[b] for b in x_) for x_ in x]
        fitnesses = []

        for sequence in sequences:
            energy = RNA.duplexfold(self._target, sequence).energy
            fitness = energy / self._norm_value
            fitnesses.append(fitness)

        return np.array(fitnesses).reshape(-1, 1)


# https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L75-L85
def _compute_min_binding_energy(target: str, design_dim: int) -> float:
    complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

    complement = "".join(complements[x] for x in target)[::-1]
    energy = RNA.duplexfold(complement, target).energy
    return energy * design_dim / len(target)


def _prepare_prompts(target: str, design_dim: int) -> tuple[str, str]:
    system_prompt = RNA_SYSTEM_PROMPT.format(design_dim=design_dim, target=target)
    return system_prompt, RNA_USER_PROMPT


def _prepare_designs(task_name: str, data_dir: Path) -> np.ndarray:
    return np.load(data_dir / f"{task_name}_x.npy")


def _render_rna_design(x: np.ndarray) -> str:
    return f"<design>{[BASES[b] for b in x]}</design>"


def _render_rna_example(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"RNA: {_render_rna_design(x)}, "
        f"Binding Score: {round(y.item(), SCORE_PRECISION)}"
    )


def _parse_rna_completion(completion: str, design_dim: int) -> tuple[list[int], bool]:
    matches = DESIGN_PATTERN.findall(completion)

    if not matches:
        return [0] * design_dim, False

    try:
        design = ast.literal_eval(matches[-1])
    except Exception:
        design = list(matches[-1].strip())

    if isinstance(design, str):
        design = list(design)

    if (
        not isinstance(design, list)
        or len(design) != design_dim
        or not all(b in BASES for b in design)
    ):
        return [0] * design_dim, False

    return [BASES.index(b) for b in design], True
