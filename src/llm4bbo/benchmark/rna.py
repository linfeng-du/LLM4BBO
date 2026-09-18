__all__ = ["RNATask"]

import RNA

import numpy as np

from .base import BenchmarkTask, register_tasks
from .prompts.rna import RNA_SYSTEM_PROMPT_TEMPLATE


# https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L137-L143
_TASK_METADATA = {
    "rna1": (
        "L14_RNA1",
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACC"
        "CCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC"
    ),
    "rna2": (
        "L14_RNA2",
        "GAGGCACAUUCCGGCUCGCCCCCGUCCGCGCGGGGGCCCCGCGCGGACGG"
        "GGUCCGGCCCGCGCGGGGCCCCCGCGCGGGAGCCGGAAUGUGCCUCGUUC"
    ),
    "rna3": (
        "L14_RNA3",
        "CCGGUGAUACUGUUAGUGGUCACGGUGCAUUUAUAGCGCUAAAGUACAGU"
        "CUUCCCCUGUUGAACGGCGCCAUUGCAUACAGGGCCAGCCGCGUAACGCC"
    )
}


@register_tasks(*_TASK_METADATA)
class RNATask(BenchmarkTask):
    benchmark: str = "rna"

    def __init__(self, task_key: str, num_designs: int) -> None:
        task_name, self._target = _TASK_METADATA[task_key]
        x_offline = np.load(self.data_dir / f"{task_name}_x.npy")

        categories = ["U", "G", "C", "A"]
        x_low = np.zeros(x_offline.shape[1], dtype=x_offline.dtype)
        x_high = np.full(x_offline.shape[1], len(categories) - 1, dtype=x_offline.dtype)

        # Compute the reference binding energy used to normalize oracle predictions
        # https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L75-L85
        complements = {"A": "U", "C": "G", "G": "C", "U": "A"}
        complement = "".join(complements[b] for b in self._target)[::-1]
        energy = RNA.duplexfold(complement, self._target).energy
        self._norm_value = energy * x_offline.shape[1] / len(self._target)

        super().__init__(
            task_name=task_name,
            num_designs=num_designs,
            system_prompt=RNA_SYSTEM_PROMPT_TEMPLATE.format(target=self._target),
            x_offline=x_offline,
            x_low=x_low,
            x_high=x_high,
            categories=categories
        )

    # https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py#L87-L116
    def predict(self, x: np.ndarray) -> np.ndarray:
        y = []

        for x_i in x:
            sequence = "".join(self.categories[i] for i in x_i)
            energy = RNA.duplexfold(self._target, sequence).energy
            y.append(energy / self._norm_value)

        return np.array(y).reshape(-1, 1)
