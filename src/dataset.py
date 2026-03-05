from pathlib import Path
from typing import Any

import wandb
from tqdm import tqdm

import patches, design_bench
from datasets import Dataset, DatasetDict
from design_bench.task import Task

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from prompt import create_prompt_fn


def sample_evenly_spaced_subset(
    task_name: str,
    subset_size: int
) -> tuple[Task, np.ndarray, np.ndarray, MinMaxScaler]:
    task, x, y, oracle_scaler = _load_relabeled_dataset(task_name)

    sorted_index = y.squeeze(axis=-1).argsort()
    spaced_index = (
        np.linspace(0, len(sorted_index) - 1, num=subset_size)
        .round().astype(int)
    )
    index = sorted_index[spaced_index]

    return task, x[index], y[index], oracle_scaler


def _load_relabeled_dataset(
    task_name: str
) -> tuple[Task, np.ndarray, np.ndarray, MinMaxScaler]:
    relabeled_dir = Path("data") / "relabeled_datasets"
    task = design_bench.make(task_name)

    # Used to normalize oracle predictions
    oracle_scaler = MinMaxScaler()

    if task_name == "TFBind10-Exact-v0":
        x = np.load(relabeled_dir / "tf10_x_correct.npy")
        y = np.load(relabeled_dir / "tf10_y_correct.npy")
        oracle_scaler.fit(y)

        # Keep the half of examples with the smallest y
        half_size = len(y) // 2
        index = y.squeeze(axis=-1).argpartition(half_size)[:half_size]
        x, y = x[index], y[index]

        # Patch `task.predict` to use relabeled y
        text = (relabeled_dir / "parsed_tf10.txt").read_text()
        table = {
            k: float(v)
            for line in text.splitlines()
            for k, v in [line.split()]
        }

        def tfbind10_predict(x: np.ndarray) -> np.ndarray:
            x_char = np.array(['A', 'C', 'G', 'T'])[x]
            return np.array([[table["".join(x)]] for x in x_char])

        task.predict = tfbind10_predict

    else:
        x = task.x
        y = np.load(relabeled_dir / f"{task_name}_relabeled_y.npy")

        # Create a temporary task object to avoid mutating `task`
        tmp_task = design_bench.make(task_name)
        tmp_task.dataset.subsample()
        oracle_scaler.fit(tmp_task.dataset.y)

    return task, x, y, oracle_scaler


def build_dataset(
    task_name: str,
    subset_size: int,
    stage: str,
    val_size: int | float,
    seed: int,
    scale_reward: bool | None = None,
    **kwargs: Any
) -> DatasetDict:
    task, x, y, _ = sample_evenly_spaced_subset(task_name, subset_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, random_state=seed
    )

    # Min-max normalize y to ensure a consistent reward scale across tasks
    scaler = MinMaxScaler()
    y_train_norm = scaler.fit_transform(y_train)
    y_val_norm = scaler.transform(y_val)

    rng = np.random.default_rng(seed)

    if stage == "sft":
        train_dataset = _build_offline_rl_dataset(
            task_name, task, x_train, y_train, y_train_norm, rng=rng, **kwargs
        )
        val_dataset = _build_offline_rl_dataset(
            task_name, task, x_val, y_val, y_val_norm, rng=rng, **kwargs
        )

        train_dataset = (
            train_dataset.filter(lambda example: example["reward"] > 0)
            .remove_columns("reward")
        )
        val_dataset = (
            val_dataset.filter(lambda example: example["reward"] > 0)
            .remove_columns("reward")
        )

    elif stage == "offline_rl":
        assert scale_reward is not None

        train_dataset = _build_offline_rl_dataset(
            task_name, task, x_train, y_train, y_train_norm, rng=rng, **kwargs
        )
        val_dataset = _build_offline_rl_dataset(
            task_name, task, x_val, y_val, y_val_norm, rng=rng, **kwargs
        )

        if scale_reward:
            # Scale rewards by inverse of the global std
            r_train_std = np.std(train_dataset["reward"]).item()
            assert r_train_std > 0

            train_dataset = train_dataset.map(
                lambda example: {"reward": example["reward"] / r_train_std}
            )
            val_dataset = val_dataset.map(
                lambda example: {"reward": example["reward"] / r_train_std}
            )

        train_ratio = np.mean(np.array(train_dataset["reward"]) > 0).item()
        val_ratio = np.mean(np.array(val_dataset["reward"]) > 0).item()

        table = wandb.Table(
            columns=["split", "positive_reward_ratio"],
            data=[["train", train_ratio], ["validation", val_ratio]]
        )
        wandb.log({
            "dataset/positive_reward_ratio": wandb.plot.bar(
                table,
                label="split",
                value="positive_reward_ratio",
                title="Positive Reward Ratio"
            )
        })

    elif stage == "online_rl":
        train_dataset = _build_online_rl_dataset(
            task_name, x_train, y_train, rng=rng, **kwargs
        )
        val_dataset = _build_online_rl_dataset(
            task_name, x_val, y_val, rng=rng, **kwargs
        )

    else:
        raise ValueError(f"Invalid stage: {stage}")

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def _build_offline_rl_dataset(
    task_name: str,
    task: Task,
    x: np.ndarray,
    y: np.ndarray,
    y_norm: np.ndarray,
    response_ratio: float,
    num_candidates: int,
    num_shots: int,
    num_permutations: int,
    rng: np.random.Generator
) -> Dataset:
    # Partition the dataset into disjoint response and prompt subsets
    index = rng.permutation(len(x))
    x, y, y_norm = x[index], y[index], y_norm[index]
    response_size = int(len(x) * response_ratio)

    x_response, y_norm_response = x[:response_size], y_norm[:response_size]
    x_prompt, y_prompt, y_norm_prompt = (
        x[response_size:], y[response_size:], y_norm[response_size:]
    )

    if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
        similarity = rbf_kernel(
            task.to_logits(x_response).reshape(len(x_response), -1),
            task.to_logits(x_prompt).reshape(len(x_prompt), -1)
        )
    elif task_name in {"AntMorphology-Exact-v0", "DKittyMorphology-Exact-v0"}:
        similarity = rbf_kernel(x_response, x_prompt)
    else:
        raise ValueError(f"Invalid task: {task_name}")

    prompt_fn = create_prompt_fn(task_name)

    examples = []

    for x_resp, y_norm_resp, sim in tqdm(
        zip(x_response, y_norm_response, similarity, strict=True),
        desc=f"Building offline RL dataset for {task_name}",
        total=len(x_response)
    ):
        # Retrieve candidates with the highest kernel-based similarity
        index = sim.argpartition(-num_candidates)[-num_candidates:]
        x_cand, y_cand, y_norm_cand = (
            x_prompt[index], y_prompt[index], y_norm_prompt[index]
        )

        worse_index = np.where(y_norm_resp > y_norm_cand)[0]

        if len(worse_index) >= num_shots:
            # Positive reward: sample from candidates worse than the response
            index = rng.choice(worse_index, size=num_shots, replace=False)
        else:
            # Negative reward: sample from all candidates
            index = rng.permutation(num_candidates)[:num_shots]

        x_ref, y_ref, y_norm_ref = (
            x_cand[index], y_cand[index], y_norm_cand[index]
        )

        reward = (y_norm_resp - y_norm_ref.max()).item()

        # Include different permutations of the references
        for _ in range(num_permutations):
            index = rng.permutation(len(x_ref))
            prompt, completion = prompt_fn(x_ref[index], y_ref[index], x_resp)
            examples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reward": reward
                }
            )

    return Dataset.from_list(examples)


def _build_online_rl_dataset(
    task_name: str,
    x: np.ndarray,
    y: np.ndarray,
    dataset_size: int,
    num_shots: int,
    rng: np.random.Generator
) -> Dataset:
    prompt_fn = create_prompt_fn(task_name)
    examples = []

    for _ in tqdm(
        range(dataset_size), desc=f"Building online RL dataset for {task_name}"
    ):
        index = rng.choice(len(x), size=num_shots, replace=False)
        x_ref, y_ref = x[index], y[index]
        examples.append(
            {
                "prompt": prompt_fn(x_ref, y_ref),
                "best_reference_score": y_ref.max().item()
            }
        )

    return Dataset.from_list(examples)
