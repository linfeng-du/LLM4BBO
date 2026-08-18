import gc
import multiprocessing as mp

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch

import llm4bbo.patches
from transformers import AutoTokenizer
from transformers.pipelines.text_generation import ChatType
from trl import GRPOConfig, GRPOTrainer

from llm4bbo.dataset import build_dataset
from llm4bbo.generate import GenerateWithBudgets
from llm4bbo.gpr import create_reward
from llm4bbo.trainer.evaluate import evaluate
from llm4bbo.trainer.utils import get_best_model, update_config


@hydra.main(config_path="config", config_name="online_rl_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    update_config(cfg)
    main_online_rl(cfg)


def main_online_rl(cfg: DictConfig) -> None:
    wandb.init(**OmegaConf.to_container(cfg.wandb_init, resolve=True))

    dataset = build_dataset(**cfg.build_dataset)

    if cfg.init_from == "base":
        model = cfg.llm.model
    else:
        model = get_best_model(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.model, padding_side="left", truncation_side="left"
    )

    trainer = GRPOTrainer(
        model,
        reward_funcs=[create_reward(cfg.task_name)],
        args=GRPOConfig(**OmegaConf.to_container(cfg.grpo_config, resolve=True)),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        rollout_func=thinking_budget_rollout_func
    )

    if cfg.grpo_config.vllm_mode == "server":
        trainer.vllm_generation.vllm_client.generate = GenerateWithBudgets(
            trainer.vllm_generation.vllm_client.generate,
            tokenizer,
            cfg.thinking_budget,
            cfg.answer_budget
        )
    elif cfg.grpo_config.vllm_mode == "colocate":
        trainer.vllm_generation.llm.generate = GenerateWithBudgets(
            trainer.vllm_generation.llm.generate,
            tokenizer,
            cfg.thinking_budget,
            cfg.answer_budget
        )

    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if cfg.grpo_config.vllm_mode == "colocate":
        OmegaConf.resolve(cfg)

        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()

        p = ctx.Process(target=evaluate, args=(cfg, results_queue))
        p.start()

        results = results_queue.get()
        p.join()

        wandb.summary.update(results["evaluate"])
        wandb.summary["evaluate/best_conversations"] = wandb.Table(**results["table"])
    else:
        evaluate(cfg)


def thinking_budget_rollout_func(
    prompts: list[ChatType],
    trainer: GRPOTrainer
) -> dict[str, list[list[int]] | list[list[float]]]:
    # https://github.com/huggingface/trl/blob/v1.10.0/trl/trainer/grpo_trainer.py#L2241
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)
    completion_ids, logprobs, _ = trainer._generate_single_turn(
        prompt_ids, images, multimodal_fields
    )

    # Mask the stop thinking tokens in `completion_ids`
    env_mask = [[1] * len(ids) for ids in completion_ids]

    if trainer.vllm_generation.mode == "server":
        generate = trainer.vllm_generation.vllm_client.generate
    elif trainer.vllm_generation.mode == "colocate":
        generate = trainer.vllm_generation.llm.generate

    for ids, mask in zip(completion_ids, env_mask, strict=True):
        if generate.eoth_token_id not in ids:
            continue

        eoth_index = ids.index(generate.eoth_token_id)
        eoth_offset = generate.stop_thinking_ids.index(generate.eoth_token_id)
        mask_start = eoth_index - eoth_offset
        mask_end = mask_start + len(generate.stop_thinking_ids)

        if ids[mask_start:mask_end] == generate.stop_thinking_ids:
            mask[mask_start:mask_end] = [0] * len(generate.stop_thinking_ids)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask
    }


if __name__ == "__main__":
    main()
