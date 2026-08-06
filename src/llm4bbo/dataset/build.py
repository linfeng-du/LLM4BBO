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

from .codec import create_prompt_fn


def prepare_task(task_name: str) -> tuple[Task, np.ndarray, np.ndarray, MinMaxScaler]:
    task = design_bench.make(task_name)

    # Fit on all available targets so outputs from `task.predict`
    # can later be normalized using the task's full target range
    oracle_scaler = MinMaxScaler()
    data_dir = resources.files("llm4bbo") / "assets" / "data"

    if task_name == "TFBind10-Exact-v0":
        x_all = np.load(data_dir / f"{task_name}_x.npy")
        y_all = np.load(data_dir / f"{task_name}_y.npy")
        oracle_scaler.fit(y_all)

        # Use the lower-scoring half of the designs as the offline dataset
        half_size = len(y_all) // 2
        half_index = y_all.squeeze(-1).argpartition(half_size)[:half_size]
        x, y = x_all[half_index], y_all[half_index]

        # Patch `task.predict` to use the relabeled targets
        text = (data_dir / f"{task_name}_oracle.txt").read_text()
        targets = {k: float(v) for line in text.splitlines() for k, v in [line.split()]}

        def predict(x_pred: np.ndarray) -> np.ndarray:
            x_char = np.array(["A", "C", "G", "T"])[x_pred]
            return np.array([[targets["".join(xc)]] for xc in x_char])

        task.predict = predict

    else:
        x = task.x
        y = np.load(data_dir / f"{task_name}_y.npy")

        # Create a temporary task object to avoid mutating `task`
        tmp_task = design_bench.make(task_name)
        tmp_task.dataset.subsample()
        oracle_scaler.fit(tmp_task.dataset.y)

    return task, x, y, oracle_scaler


def select_evenly_spaced(y: np.ndarray, num_designs: int) -> np.ndarray:
    sorted_index = y.squeeze(-1).argsort()
    spaced_index = np.linspace(0, len(y) - 1, num_designs).round().astype(int)
    return sorted_index[spaced_index]


def build_dataset(
    task_name: str,
    stage: str,
    num_designs: int,
    val_ratio: float,
    seed: int,
    **kwargs: Any
) -> DatasetDict:
    assert stage in {"trace", "sft", "offline_rl", "online_rl"}

    task, x, y, _ = prepare_task(task_name)
    sample_index = select_evenly_spaced(y, num_designs)
    x_sample, y_sample = x[sample_index], y[sample_index]

    x_train, x_val, y_train, y_val = train_test_split(
        x_sample, y_sample, test_size=val_ratio, random_state=seed
    )

    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)

    if stage == "online_rl":
        train_dataset = _build_online_dataset(
            task_name, x_train, y_train, train_rng, **kwargs
        )
        val_dataset = _build_online_dataset(
            task_name, x_val, y_val, val_rng, **kwargs
        )
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Normalize the targets to keep reward scales consistent across tasks
    scaler = MinMaxScaler()
    y_train_norm = scaler.fit_transform(y_train)
    y_val_norm = scaler.transform(y_val)

    if stage == "offline_rl":
        scale_reward = kwargs.pop("scale_reward")

    train_dataset = _build_offline_dataset(
        task_name, task, x_train, y_train, y_train_norm,
        stage == "trace", train_rng, **kwargs
    )
    val_dataset = _build_offline_dataset(
        task_name, task, x_val, y_val, y_val_norm,
        stage == "trace", val_rng, **kwargs
    )

    if stage in {"trace", "sft"}:
        is_positive = lambda example: example["reward"] > 0
        train_dataset = train_dataset.filter(is_positive).remove_columns("reward")
        val_dataset = val_dataset.filter(is_positive).remove_columns("reward")

        if stage == "sft":
            # Add metadata read by `SFTTrainer` to disable thinking during SFT
            disable_thinking = lambda example: {
                "chat_template_kwargs": {"enable_thinking": False}
            }
            train_dataset = train_dataset.map(disable_thinking)
            val_dataset = val_dataset.map(disable_thinking)

    elif scale_reward:
        # Divide the rewards by the training-set standard deviation
        r_train_std = np.std(train_dataset["reward"]).item()
        assert r_train_std > 0

        divide_reward = lambda example: {"reward": example["reward"] / r_train_std}
        train_dataset = train_dataset.map(divide_reward)
        val_dataset = val_dataset.map(divide_reward)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def _build_offline_dataset(
    task_name: str,
    task: Task,
    x: np.ndarray,
    y: np.ndarray,
    y_norm: np.ndarray,
    generate_trace: bool,
    rng: np.random.Generator,
    response_ratio: float,
    candidate_strategy: str,
    num_candidates: int,
    num_permutations: int,
    num_shots: int,
    use_tools: bool
) -> Dataset:
    # Partition the designs into disjoint response and prompt subsets
    perm_index = rng.permutation(len(x))
    x_perm, y_perm, y_norm_perm = x[perm_index], y[perm_index], y_norm[perm_index]
    response_size = int(len(x_perm) * response_ratio)

    x_response, y_norm_response = (
        x_perm[:response_size], y_norm_perm[:response_size]
    )
    x_prompt, y_prompt, y_norm_prompt = (
        x_perm[response_size:], y_perm[response_size:], y_norm_perm[response_size:]
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
    prompt_fn = create_prompt_fn(task_name)

    for i, (x_resp, y_norm_resp) in tqdm(
        enumerate(zip(x_response, y_norm_response, strict=True)),
        desc="Building offline dataset",
        total=len(x_response)
    ):
        if candidate_strategy == "random":
            cand_index = rng.choice(len(x_prompt), num_candidates, replace=False)
        elif candidate_strategy == "similarity":
            # Select the candidates with the highest kernel-based similarity scores
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

        # Include different permutations of the reference designs
        for _ in range(num_permutations):
            ref_perm_index = rng.permutation(len(x_ref))
            prompt, completion = prompt_fn(
                x_ref[ref_perm_index], y_ref[ref_perm_index], use_tools,
                x_resp, generate_trace
            )

            examples.append({
                "prompt": prompt,
                "completion": completion,
                "reward": reward
            })

    return Dataset.from_list(examples)


def _build_online_dataset(
    task_name: str,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    dataset_size: int,
    num_shots: int,
    use_tools: bool
) -> Dataset:
    examples = []
    prompt_fn = create_prompt_fn(task_name)

    for _ in tqdm(range(dataset_size), desc="Building online dataset"):
        ref_index = rng.choice(len(x), num_shots, replace=False)
        x_ref, y_ref = x[ref_index], y[ref_index]

        prompt = prompt_fn(x_ref, y_ref, use_tools)
        best_f = y_ref.max().item()
        examples.append({"prompt": prompt, "best_f": best_f})

    return Dataset.from_list(examples)
