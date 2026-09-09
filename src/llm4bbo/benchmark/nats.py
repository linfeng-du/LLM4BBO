__all__ = ["NATSBenchTask"]

from itertools import product

import nats_bench
import numpy as np

from .base import BenchmarkTask, register_tasks
from .prompts.nats import NATS_SYSTEM_PROMPTS


_TASK_METADATA = {
    "tss10": ("tss", "cifar10"),
    "tss100": ("tss", "cifar100"),
    "tss120": ("tss", "ImageNet16-120"),
    "sss10": ("sss", "cifar10"),
    "sss100": ("sss", "cifar100"),
    "sss120": ("sss", "ImageNet16-120")
}

_ARCHIVE_NAMES = {
    "tss": "NATS-tss-v1_0-3ffb9-simple",
    "sss": "NATS-sss-v1_0-50262-simple"
}


@register_tasks(*_TASK_METADATA)
class NATSBenchTask(BenchmarkTask):
    benchmark: str = "nats_bench"
    score_precision: int = 3

    def __init__(self, task_key: str, num_designs: int) -> None:
        self._search_space, self._dataset = _TASK_METADATA[task_key]
        task_name = f"{self._search_space}-{self._dataset}"
        self._info = nats_bench.search_space_info("nats-bench", self._search_space)

        self._api = nats_bench.create(
            str(self.benchmark_dir / _ARCHIVE_NAMES[self._search_space]),
            self._search_space,
            fast_mode=True
        )

        if self._search_space == "tss":
            op_names = self._info["op_names"]
            num_nodes = self._info["num_nodes"]

            values = list(range(len(op_names)))
            design_dim = num_nodes * (num_nodes - 1) // 2
            categories = op_names
            allowed_values = None
        else:
            values = self._info["candidates"]
            design_dim = self._info["num_layers"]
            categories = None
            allowed_values = values

        # Compute scores for all designs
        x_all = np.array(list(product(values, repeat=design_dim)))
        cache_path = self.data_dir / f"{task_name}_y.npy"
        y_all = self.predict(x_all, cache_path=cache_path)

        # Use designs in the lower 50th percentile as the offline dataset
        size = len(y_all) // 2
        indices = y_all.squeeze(-1).argpartition(size)[:size]
        x_offline = x_all[indices]

        super().__init__(
            task_name=task_name,
            num_designs=num_designs,
            system_prompt=NATS_SYSTEM_PROMPTS[self._search_space, self._dataset],
            x_offline=x_offline,
            categories=categories,
            allowed_values=allowed_values
        )

    def _predict(self, x: np.ndarray) -> np.ndarray:
        design_dim = 6 if self._search_space == "tss" else 5
        if x.ndim != 2 or x.shape[1] != design_dim:
            raise ValueError(
                f"x must have shape (n, {design_dim}), got {x.shape}"
            )

        if x.dtype.kind not in "iuf" or not np.all(np.isin(x, self._values)):
            raise ValueError(f"Each parameter must be one of {self._values}")

        scores = []
        for design in x:
            if self._search_space == "tss":
                # Edge order: 0->1, 0->2, 1->2, 0->3, 1->3, 2->3.
                operations = [_OPERATIONS[int(p)] for p in design]
                architecture = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
                    *operations
                )
            else:
                architecture = ":".join(str(int(p)) for p in design)

            # Resolve the official index; Cartesian-product order need not match it.
            index = self._task.query_index_by_arch(architecture)
            if index < 0:
                raise ValueError(f"Architecture not found in NATS-Bench: {architecture}")

            info = self._task.get_more_info(
                index, self._dataset, hp=self._hp, is_random=False
            )
            scores.append(info["test-accuracy"])

        return np.array(scores, dtype=float).reshape(-1, 1)
