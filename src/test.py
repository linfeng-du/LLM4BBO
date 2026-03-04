import json
import logging
import os

# Prevent `transformers` from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig

import numpy as np
from vllm import LLM, SamplingParams

from dataset import sample_evenly_spaced_subset
from prompt import create_parse_fn, create_prompt_fn
from resolver import register_resolvers
from thinking_budget import ThinkingBudgetVLLMGeneration


logger = logging.getLogger(__name__)
register_resolvers()


@hydra.main(config_path="../conf", config_name="test", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    task, x, y, oracle_scaler = sample_evenly_spaced_subset(
        cfg.task_name, cfg.subset_size
    )

    llm = LLM(cfg.model)
    tokenizer = llm.get_tokenizer()
    vllm_generation = ThinkingBudgetVLLMGeneration(
        llm, tokenizer, cfg.test.thinking_budget, cfg.llm.eoth_token
    )

    prompt_fn = create_prompt_fn(cfg.task_name)
    parse_fn = create_parse_fn(cfg.task_name)

    chat_prompts = []

    for seed in range(cfg.test.num_proposals):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(x), size=cfg.test.num_shots, replace=False)
        chat_prompts.append(prompt_fn(x[indices], y[indices]))

    prompts = tokenizer.apply_chat_template(
        chat_prompts, add_generation_prompt=True, tokenize=False
    )
    sampling_params = SamplingParams(
        n=cfg.test.num_trials,
        **cfg.llm.generation_kwargs
    )
    requests = vllm_generation(prompts, sampling_params)
    completions = [o.text for r in requests for o in r.outputs]

    x_pred = parse_fn(completions)
    y_pred = (
        oracle_scaler.transform(task.predict(x_pred))
        .reshape(cfg.test.num_trials, cfg.test.num_proposals, order="F")
    )
    y_pred_max = y_pred.max(axis=-1)
    y_pred_median = np.median(y_pred, axis=-1)

    results = {
        "max_mean": y_pred_max.mean().item(),
        "max_std": y_pred_max.std().item(),
        "median_mean": y_pred_median.mean().item(),
        "median_std": y_pred_median.std().item()
    }
    logger.info(f"\n{json.dumps(results, indent=4)}")
    wandb.log({f"test/{k}": v for k, v in results.items()})

    table = wandb.Table(columns=["trial", "system", "user", "assistant"])
    chat_strs = []

    for trial_index, proposal_index in enumerate(y_pred.argmax(axis=-1)):
        chat_prompt = chat_prompts[proposal_index]
        completion = completions[
            proposal_index * cfg.test.num_trials + trial_index
        ]
        chat_strs.append(
            f"[SYSTEM PROMPT]\n{chat_prompt[0]['content']}\n\n"
            f"[USER PROMPT]\n{chat_prompt[1]['content']}\n\n"
            f"[ASSISTANT COMPLETION]\n{completion}\n\n"
        )
        table.add_data(
            trial_index,
            chat_prompt[0]["content"],
            chat_prompt[1]["content"],
            completion
        )

    chat_str = f"\n{'-' * 100}\n".join(chat_strs)
    logger.info(f"\n{chat_str}")
    wandb.log({"test/chats": table})


if __name__ == "__main__":
    main()
