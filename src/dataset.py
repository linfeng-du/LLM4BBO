import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from datasets import Dataset, DatasetDict

import patches
import design_bench
from design_bench.task import Task

from prompt import create_prompt_fn


logger = logging.getLogger(__name__)


def load_evenly_spaced_examples(
    task_name: str,
    dataset_size: int
) -> tuple[Task, np.ndarray, np.ndarray]:
    task = design_bench.make(task_name)

    x = task.x
    y = _load_relabeled_y(task_name, task)

    sorted_indices = y.squeeze(axis=-1).argsort()
    spaced_indices = (
        np.linspace(0, len(sorted_indices) - 1, num=dataset_size)
        .round().astype(int)
    )

    indices = sorted_indices[spaced_indices]
    return task, x[indices], y[indices]


def _load_relabeled_y(task_name: str, task: Task) -> np.ndarray:
    relabeled_y_file = Path("data") / f"{task_name}_relabeled_y.npy"

    if not relabeled_y_file.exists():
        logger.info(f"Relabeling {task_name}...")
        y_relabeled = task.predict(task.x)
        np.save(relabeled_y_file, y_relabeled)

    return np.load(relabeled_y_file)


def build_rl_dataset(
    task_name: str,
    dataset_size: int,
    val_size: int,
    random_state: int,
    mode: str,
    **kwargs: dict[str, Any]
) -> DatasetDict:
    task, x, y = load_evenly_spaced_examples(task_name, dataset_size)

    # Min-max normalize `y` to ensure consistent reward scale across tasks
    scaler = MinMaxScaler()
    y_norm = scaler.fit_transform(y)

    x_train, x_val, y_train, y_val, y_norm_train, y_norm_val = (
        train_test_split(
            x, y, y_norm, test_size=val_size, random_state=random_state
        )
    )

    rng = np.random.default_rng(random_state)

    if mode == "offline":
        train_dataset = _build_offline_rl_dataset(
            task_name, task, x_train, y_train, y_norm_train, rng, **kwargs
        )
        val_dataset = _build_offline_rl_dataset(
            task_name, task, x_val, y_val, y_norm_val, rng, **kwargs
        )

        # Normalize rewards by global standard deviation
        r_train_std = np.std(train_dataset["reward"])
        train_dataset = train_dataset.map(
            lambda example: {"reward": example["reward"] / r_train_std}
        )
        val_dataset = val_dataset.map(
            lambda example: {"reward": example["reward"] / r_train_std}
        )

    elif mode == "online":
        train_dataset = _build_online_rl_dataset(
            task_name, x_train, y_train, rng, **kwargs
        )
        val_dataset = _build_online_rl_dataset(
            task_name, x_val, y_val, rng, **kwargs
        )

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def _build_offline_rl_dataset(
    task_name: str,
    task: Task,
    x: np.ndarray,
    y: np.ndarray,
    y_norm: np.ndarray,
    rng: np.random.Generator,
    response_fraction: float,
    num_candidates: int,
    num_shots: int,
    num_permutations: int
) -> Dataset:
    indices = rng.permutation(len(x))
    x, y, y_norm = x[indices], y[indices], y_norm[indices]

    # Partition the dataset into disjoint response and prompt subsets
    response_size = int(len(x) * response_fraction)
    assert len(x) - response_size >= num_candidates

    x_responses, y_norm_responses = x[:response_size], y_norm[:response_size]
    x_prompt, y_prompt, y_norm_prompt = (
        x[response_size:], y[response_size:], y_norm[response_size:]
    )

    if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
        similarity_matrix = rbf_kernel(
            task.to_logits(x_responses).reshape(len(x_responses), -1),
            task.to_logits(x_prompt).reshape(len(x_prompt), -1)
        )

    elif task_name in {"AntMorphology-Exact-v0", "DKittyMorphology-Exact-v0"}:
        similarity_matrix = rbf_kernel(
            x_responses.reshape(len(x_responses), -1),
            x_prompt.reshape(len(x_prompt), -1)
        )

    else:
        raise ValueError(f"Invalid task: {task_name}")

    prompt_fn = create_prompt_fn(task_name)
    examples = []

    for x_response, y_norm_response, similarities in tqdm(
        zip(x_responses, y_norm_responses, similarity_matrix, strict=True),
        desc=f"Building offline RL dataset for {task_name}",
        total=len(x_responses)
    ):
        # Retrieve candidates with the highest kernel-based similarity
        indices = similarities.argpartition(-num_candidates)[-num_candidates:]
        x_candidates, y_candidates, y_norm_candidates = (
            x_prompt[indices], y_prompt[indices], y_norm_prompt[indices]
        )

        worse_indices = np.where(y_norm_response > y_norm_candidates)[0]

        if len(worse_indices) >= num_shots:
            # Sample only from candidates that are worse than the response
            indices = rng.choice(worse_indices, size=num_shots, replace=False)
        else:
            indices = rng.permutation(num_candidates)[:num_shots]

        x_references, y_references, y_norm_references = (
            x_candidates[indices],
            y_candidates[indices],
            y_norm_candidates[indices]
        )

        # Compute reward as improvement from the references to the response
        reward = (y_norm_response - y_norm_references.max()).item()

        # Include different permutations of the references
        for _ in range(num_permutations):
            indices = rng.permutation(len(x_references))
            prompt, completion = prompt_fn(
                x_references[indices],
                y_references[indices],
                x_response=x_response
            )
            examples.append({
                "prompt": prompt, "completion": completion, "reward": reward
            })

    return Dataset.from_list(examples)


def _build_online_rl_dataset(
    task_name: str,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    online_dataset_size: int,
    num_shots: int
) -> Dataset:
    prompt_fn = create_prompt_fn(task_name)
    examples = []

    for _ in tqdm(
        range(online_dataset_size),
        desc=f"Building online RL dataset for {task_name}"
    ):
        indices = rng.choice(len(x), size=num_shots, replace=False)
        x_references, y_references = x[indices], y[indices]
        examples.append({
            "prompt": prompt_fn(x_references, y_references),
            "best_reference_score": y_references.max().item()
        })

    return Dataset.from_list(examples)
