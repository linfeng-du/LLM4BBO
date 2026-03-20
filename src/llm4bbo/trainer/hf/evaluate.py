import os
os.environ["USE_TF"] = "0"

import json
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import numpy as np
from vllm import LLM, SamplingParams

from llm4bbo.dataset import (
    create_parse_fn,
    create_prompt_fn,
    sample_evenly_spaced_subset
)
from llm4bbo.trainer.hf.thinking_budget import ThinkingBudgetVLLMGenerate
from llm4bbo.trainer.hf.utils import get_model, update_config


@hydra.main(config_path="config", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    update_config(cfg)
    wandb.init(**OmegaConf.to_container(cfg.wandb_init, resolve=True))
    evaluate(cfg)


def evaluate(cfg: DictConfig) -> None:
    task, x, y, oracle_scaler = sample_evenly_spaced_subset(
        cfg.task_name, cfg.subset_size
    )

    if cfg.stage == "base":
        model = cfg.llm.model
    else:
        model = get_model(cfg.output_dir)

    llm = LLM(model)
    tokenizer = llm.get_tokenizer()
    vllm_generate = ThinkingBudgetVLLMGenerate(
        llm, tokenizer, cfg.evaluate.thinking_budget
    )

    prompt_fn = create_prompt_fn(cfg.task_name)
    parse_fn = create_parse_fn(cfg.task_name)

    chat_prompts = []

    for seed in range(cfg.evaluate.num_proposals):
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            len(x), size=cfg.evaluate.num_shots, replace=False
        )
        chat_prompts.append(prompt_fn(x[indices], y[indices]))

    prompts = tokenizer.apply_chat_template(
        chat_prompts, add_generation_prompt=True, tokenize=False
    )
    sampling_params = SamplingParams(
        n=cfg.evaluate.num_trials,
        **cfg.llm.sampling_params
    )

    requests = vllm_generate(prompts, sampling_params)
    completions = [o.text for r in requests for o in r.outputs]

    x_pred = parse_fn(completions)
    y_pred = (
        oracle_scaler.transform(task.predict(x_pred))
        .reshape(
            cfg.evaluate.num_trials, cfg.evaluate.num_proposals, order="F"
        )
    )

    y_pred_max = y_pred.max(axis=-1)
    y_pred_median = np.median(y_pred, axis=-1)

    results = {
        "max_mean": y_pred_max.mean().item(),
        "max_std": y_pred_max.std().item(),
        "median_mean": y_pred_median.mean().item(),
        "median_std": y_pred_median.std().item()
    }
    output_dir = Path(cfg.output_dir)

    (output_dir / "evaluate.json").write_text(json.dumps(results, indent=2))
    wandb.summary.update({f"evaluate/{k}": v for k, v in results.items()})

    best_conversations = []
    table = wandb.Table(columns=["trial", "system", "user", "assistant"])

    for trial_index, proposal_index in enumerate(y_pred.argmax(axis=-1)):
        chat_prompt = chat_prompts[proposal_index]
        completion = completions[
            proposal_index * cfg.evaluate.num_trials + trial_index
        ]

        best_conversations.append({
            "trial": trial_index,
            "system": chat_prompt[0]["content"],
            "user": chat_prompt[1]["content"],
            "assistant": completion,
        })
        table.add_data(
            trial_index,
            chat_prompt[0]["content"],
            chat_prompt[1]["content"],
            completion
        )

    (output_dir / "best_conversations.json").write_text(
        json.dumps(best_conversations, indent=2)
    )
    wandb.summary["evaluate/best_conversations"] = table


if __name__ == "__main__":
    main()
