from tqdm import tqdm

import design_bench
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from prompt import generate_completion_tfbind8, generate_prompt_tfbind8


def build_offline_dataset(
    task_name: str,
    train_size: int,
    val_size: int,
    response_ratio: float,
    num_references: int,
    num_permutations: int,
    seed: int
) -> DatasetDict:
    task = design_bench.make(task_name)

    x_train, x_val, y_train, y_val = train_test_split(
        task.x, task.y,
        test_size=val_size, train_size=train_size, random_state=seed
    )

    # Min-max normalize target values
    y_train_min, y_train_max = (y_train.min(), y_train.max())
    y_train = (y_train - y_train_min) / (y_train_max - y_train_min)
    y_val = (y_val - y_train_min) / (y_train_max - y_train_min)

    rng = np.random.default_rng(seed)

    train_dataset = _build_offline_dataset_split(
        x_train, y_train, task_name, 'train',
        response_ratio, num_references, num_permutations, rng
    )
    val_dataset = _build_offline_dataset_split(
        x_val, y_val, task_name, 'val',
        response_ratio, num_references, num_permutations=1, rng=rng
    )

    # Divide by standard deviation of train rewards
    reward_std = np.array(train_dataset['reward']).std()

    def normalize_reward(example: dict[str, str | float]) -> dict[str, float]:
        return {'reward': example['reward'] / reward_std}

    train_dataset = train_dataset.map(normalize_reward)
    val_dataset = val_dataset.map(normalize_reward)
    return DatasetDict({'train': train_dataset, 'val': val_dataset})


def _build_offline_dataset_split(
    x: np.ndarray,
    y: np.ndarray,
    task_name: str,
    split: str,
    response_ratio: float,
    num_references: int,
    num_permutations: int,
    rng: np.random.Generator
) -> Dataset:
    indices = rng.permutation(len(x))
    x, y = (x[indices], y[indices])

    # Partition dataset into disjoint response and prompt subsets
    response_size = int(len(x) * response_ratio)
    assert 0 < response_size <= len(x) - num_references, (
        'Response size must be in (0, len(x) - num_references]'
    )

    x_response, x_prompt = (x[:response_size], x[response_size:])
    y_response, y_prompt = (y[:response_size], y[response_size:])

    # For each response, retrieve references with high kernel-based similarity
    if task_name.startswith('TFBind8'):
        # Hamming distance
        neigh = NearestNeighbors(n_neighbors=num_references, metric='hamming')
        neigh.fit(x_prompt)
        indices = neigh.kneighbors(x_response, return_distance=False)
    else:
        raise NotImplementedError(f'Similarity metric not implemented for {task_name}')

    x_references, y_references = (x_prompt[indices], y_prompt[indices])

    data = []

    for x_resp, y_resp, x_refs, y_refs in tqdm(
        zip(x_response, y_response, x_references, y_references),
        desc=f'Building offline {split} set',
        total=len(x_response)
    ):
        # Compute reward as improvement from references to response
        reward = (y_resp - y_refs.max()).item()

        # Include different permutations of references
        for _ in range(num_permutations):
            indices = rng.permutation(len(x_refs))
            x_refs_perm, y_refs_perm = (x_refs[indices], y_refs[indices])

            if task_name.startswith('TFBind8'):
                prompt = generate_prompt_tfbind8(x_refs_perm, y_refs_perm)
                completion = generate_completion_tfbind8(x_resp)

            data.append({'prompt': prompt, 'completion': completion, 'reward': reward})

    return Dataset.from_list(data)
