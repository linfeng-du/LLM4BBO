from importlib import resources
from typing import Any

from tqdm import tqdm

import llm4bbo.patches
import design_bench
from design_bench.task import Task

from datasets import Dataset, DatasetDict

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .llm_io import create_prompt_fn


def load_task_data(task_name: str) -> tuple[Task, np.ndarray, np.ndarray, MinMaxScaler]:
    dataset_dir = resources.files("llm4bbo") / "assets" / "datasets"
    task = design_bench.make(task_name)

    # Fitted on full targets; use it to normalize `task.predict` outputs
    oracle_scaler = MinMaxScaler()

    if task_name == "TFBind10-Exact-v0":
        x = np.load(dataset_dir / f"{task_name}_x.npy")
        y = np.load(dataset_dir / f"{task_name}_y.npy")
        oracle_scaler.fit(y)

        # Keep half of the designs with the smallest y
        half_size = len(y) // 2
        index = y.squeeze(-1).argpartition(half_size)[:half_size]
        x, y = x[index], y[index]

        # Patch `task.predict` to use relabeled y
        text = (dataset_dir / f"{task_name}_oracle.txt").read_text()
        oracle = {k: float(v) for line in text.splitlines() for k, v in [line.split()]}

        def predict(x: np.ndarray) -> np.ndarray:
            x_char = np.array(["A", "C", "G", "T"])[x]
            return np.array([[oracle["".join(xc)]] for xc in x_char])

        task.predict = predict

    else:
        x = task.x
        y = np.load(dataset_dir / f"{task_name}_y.npy")

        # Create a temporary task object to avoid mutating `task`
        tmp_task = design_bench.make(task_name)
        tmp_task.dataset.subsample()
        oracle_scaler.fit(tmp_task.dataset.y)

    return task, x, y, oracle_scaler


def evenly_spaced_indices(y: np.ndarray, num_designs: int) -> np.ndarray:
    sorted_index = y.squeeze(-1).argsort()
    spaced_index = np.linspace(0, len(y) - 1, num_designs).round().astype(int)
    return sorted_index[spaced_index]


def build_dataset(
    task_name: str,
    stage: str,
    num_designs: int,
    val_design_ratio: float,
    seed: int,
    **kwargs: Any
) -> DatasetDict:
    task, x, y, _ = load_task_data(task_name)
    index = evenly_spaced_indices(y, num_designs)
    x, y = x[index], y[index]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_design_ratio, random_state=seed
    )

    # Min-max normalize y to ensure a consistent reward scale across tasks
    scaler = MinMaxScaler()
    y_train_norm = scaler.fit_transform(y_train)
    y_val_norm = scaler.transform(y_val)

    rng = np.random.default_rng(seed)

    if stage == "sft":
        off_train_dataset = _build_offline_rl_dataset(
            task_name, task, x_train, y_train, y_train_norm, rng=rng, **kwargs
        )
        off_val_dataset = _build_offline_rl_dataset(
            task_name, task, x_val, y_val, y_val_norm, rng=rng, **kwargs
        )

        filter_fn = lambda example: example["reward"] > 0
        train_dataset = off_train_dataset.filter(filter_fn).remove_columns("reward")
        val_dataset = off_val_dataset.filter(filter_fn).remove_columns("reward")

    elif stage == "offline_rl":
        scale_reward = kwargs.pop("scale_reward")

        train_dataset = _build_offline_rl_dataset(
            task_name, task, x_train, y_train, y_train_norm, rng=rng, **kwargs
        )
        val_dataset = _build_offline_rl_dataset(
            task_name, task, x_val, y_val, y_val_norm, rng=rng, **kwargs
        )

        if scale_reward:
            # Divide rewards by std
            r_train_std = np.std(train_dataset["reward"]).item()
            assert r_train_std > 0

            map_fn = lambda example: {"reward": example["reward"] / r_train_std}
            train_dataset = train_dataset.map(map_fn)
            val_dataset = val_dataset.map(map_fn)

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
    enable_tools: bool,
    response_ratio: float,
    candidate_strategy: str,
    num_candidates: int,
    num_permutations: int,
    num_shots: int,
    rng: np.random.Generator
) -> Dataset:
    # Partition the dataset into disjoint response and prompt subsets
    perm_index = rng.permutation(len(x))
    x, y, y_norm = x[perm_index], y[perm_index], y_norm[perm_index]

    response_size = int(len(x) * response_ratio)

    x_response, y_norm_response = x[:response_size], y_norm[:response_size]
    x_prompt, y_prompt, y_norm_prompt = (
        x[response_size:], y[response_size:], y_norm[response_size:]
    )

    if candidate_strategy == "similarity":
        if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
            similarity = rbf_kernel(
                task.to_logits(x_response).reshape(len(x_response), -1),
                task.to_logits(x_prompt).reshape(len(x_prompt), -1)
            )
        elif task_name in {"AntMorphology-Exact-v0", "DKittyMorphology-Exact-v0"}:
            similarity = rbf_kernel(x_response, x_prompt)
        else:
            raise ValueError(f"Invalid task: {task_name}")

    examples = []
    prompt_fn = create_prompt_fn(task_name, enable_tools)

    for i, (x_resp, y_norm_resp) in tqdm(
        enumerate(zip(x_response, y_norm_response, strict=True)),
        desc="Building SFT / offline RL dataset",
        total=len(x_response)
    ):
        if candidate_strategy == "random":
            cand_index = rng.choice(len(x_prompt), num_candidates, replace=False)
        elif candidate_strategy == "similarity":
            # Retrieve candidates with the highest kernel-based similarity
            cand_index = similarity[i].argpartition(-num_candidates)[-num_candidates:]
        else:
            raise ValueError(f"Invalid candidate strategy: {candidate_strategy}")

        x_cand, y_cand, y_norm_cand = (
            x_prompt[cand_index], y_prompt[cand_index], y_norm_prompt[cand_index]
        )

        worse_index = np.where(y_norm_resp > y_norm_cand)[0]

        if len(worse_index) >= num_shots:
            # Positive reward: sample from candidates worse than the response
            ref_index = rng.choice(worse_index, num_shots, replace=False)
        else:
            # Negative reward: sample from all candidates
            ref_index = rng.permutation(num_candidates)[:num_shots]

        x_ref, y_ref, y_norm_ref = (
            x_cand[ref_index], y_cand[ref_index], y_norm_cand[ref_index]
        )
        reward = (y_norm_resp - y_norm_ref.max()).item()

        # Include different permutations of the references
        for _ in range(num_permutations):
            perm_index = rng.permutation(len(x_ref))
            prompt, completion = prompt_fn(x_ref[perm_index], y_ref[perm_index], x_resp)

            examples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )

    return Dataset.from_list(examples)


def _build_online_rl_dataset(
    task_name: str,
    x: np.ndarray,
    y: np.ndarray,
    enable_tools: bool,
    dataset_size: int,
    num_shots: int,
    rng: np.random.Generator
) -> Dataset:
    examples = []
    prompt_fn = create_prompt_fn(task_name, enable_tools)

    for _ in tqdm(range(dataset_size), desc="Building online RL dataset"):
        index = rng.choice(len(x), num_shots, replace=False)
        x_ref, y_ref = x[index], y[index]

        prompt = prompt_fn(x_ref, y_ref)
        best_f = y_ref.max().item()
        examples.append({"prompt": prompt, "best_f": best_f})

    return Dataset.from_list(examples)
