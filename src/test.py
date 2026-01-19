import json
import logging
import random

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np

import torch
from transformers import TextGenerationPipeline, pipeline

from dataset import load_evenly_spaced_examples
from prompt import create_parse_fn, create_prompt_fn


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def test_llm(cfg: DictConfig):
    pipe = pipeline(
        task="text-generation",
        model=cfg.llm.model,
        device_map="auto",
        dtype="bfloat16",
        **cfg.llm.hf_generation_kwargs
    )
    test(pipe, cfg)


def test(pipe: TextGenerationPipeline, cfg: DictConfig) -> None:
    task, x, y = load_evenly_spaced_examples(cfg.task_name, cfg.dataset_size)

    # Ensure correct normalization of `y_design`
    task.dataset.subsample()
    y_max, y_min = task.dataset.y.max(), task.dataset.y.min()

    prompt_fn = create_prompt_fn(cfg.task_name)
    parse_fn = create_parse_fn(cfg.task_name)

    y_design_maxs = []
    y_design_medians = []
    best_conversations = []

    # Run 8 trials, each generating 128 designs
    for seed in range(cfg.num_trials):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        indices = np.random.choice(len(x), size=cfg.num_designs, replace=False)
        x_references, y_references = x[indices], y[indices]
        prompt = prompt_fn(x_references, y_references)

        completions = []
        x_designs = []

        batch_size = 4
        assert cfg.num_designs % batch_size == 0

        with tqdm(total=cfg.num_designs, desc="Generating designs") as pbar:
            for _ in range(cfg.num_designs // batch_size):
                outputs = pipe(
                    prompt,
                    return_full_text=False,
                    num_return_sequences=batch_size
                )
                batch_completions = [o["generated_text"] for o in outputs]
                batch_x_design = parse_fn(batch_completions)

                completions.extend(batch_completions)
                x_designs.append(batch_x_design)
                pbar.update(n=batch_size)

        x_design = np.vstack(x_designs)
        y_design = task.predict(x_design)

        y_design_max = (y_design.max() - y_min) / (y_max - y_min)
        y_design_median = (np.median(y_design) - y_min) / (y_max - y_min)
        best_conversation = prompt + [{
            "role": "assistant", "content": completions[y_design.argmax()]
        }]

        y_design_maxs.append(y_design_max)
        y_design_medians.append(y_design_median)
        best_conversations.append(best_conversation)

    results = {
        "max_mean": np.mean(y_design_maxs).item(),
        "max_std": np.std(y_design_maxs).item(),
        "median_mean": np.mean(y_design_medians).item(),
        "median_std": np.std(y_design_medians).item(),
    }

    logger.info(json.dumps(results, indent=4))
    logger.info(json.dumps(best_conversations, indent=4))


if __name__ == "__main__":
    test_llm()
