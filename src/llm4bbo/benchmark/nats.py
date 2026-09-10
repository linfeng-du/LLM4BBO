__all__ = ["NATSBenchTask"]

from typing import Literal

import nats_bench
from nats_bench.genotype_utils import topology_str2structure

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

        # Gather all designs
        x_all = []

        for i in range(len(self._api)):
            architecture = self._api.arch(i)

            if self._search_space == "tss":
                structure = topology_str2structure(architecture)
                x_all.append([
                    self._info["op_names"].index(op)
                    for node in structure.nodes
                    for op, _ in node
                ])
            else:
                x_all.append(list(map(int, architecture.split(":"))))

        x_all = np.array(x_all)

        # Compute scores for all designs
        cache_path = self.data_dir / f"{task_name}_y.npy"
        y_all = self._cached_parallel_predict(x_all, cache_path)

        # Use designs in the lower 50th percentile as the offline dataset
        size = len(y_all) // 2
        indices = y_all.squeeze(-1).argpartition(size)[:size]
        x_offline = x_all[indices]

        if self._search_space == "tss":
            categories = self._info["op_names"]
            allowed_values = None
        else:
            categories = None
            allowed_values = self._info["candidates"]

        super().__init__(
            task_name=task_name,
            num_designs=num_designs,
            system_prompt=NATS_SYSTEM_PROMPTS[self._search_space, self._dataset],
            x_offline=x_offline,
            categories=categories,
            allowed_values=allowed_values
        )

    # Following the NATS-Bench paper,
    # use validation accuracy for optimization and test accuracy for final evaluation
    def predict(
        self,
        x: np.ndarray,
        split: Literal["valid", "test"] = "valid"
    ) -> np.ndarray:
        dataset = self._dataset

        if dataset == "cifar10" and split == "valid":
            dataset = "cifar10-valid"

        y = []

        for x_i in x:
            if self._search_space == "tss":
                architecture = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
                    *(self._info["op_names"][i] for i in x_i)
                )
            else:
                architecture = "{}:{}:{}:{}:{}".format(*x_i)

            results = self._api.get_more_info(
                self._api.query_index_by_arch(architecture),
                dataset,
                hp=self._api.full_train_epochs,
                is_random=False
            )
            y.append(results[f"{split}-accuracy"])

        return np.array(y).reshape(-1, 1)

    def _evaluate_designs(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x, split="test")
