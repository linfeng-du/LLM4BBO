import os
os.environ["USE_TF"] = "0"

import gc
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
import torch.nn as nn

from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from llm4bbo.dataset import build_dataset
from llm4bbo.trainer.evaluate import evaluate
from llm4bbo.trainer.utils import get_best_model, update_config


@hydra.main(
    config_path="config", config_name="offline_rl_trainer", version_base=None
)
def main(cfg: DictConfig) -> None:
    update_config(cfg)
    main_offline_rl(cfg)


def main_offline_rl(cfg: DictConfig) -> None:
    wandb.init(**OmegaConf.to_container(cfg.wandb_init, resolve=True))

    dataset = build_dataset(**cfg.build_dataset)

    train_ratio = np.mean(np.array(dataset["train"]["reward"]) > 0).item()
    val_ratio = np.mean(np.array(dataset["validation"]["reward"]) > 0).item()

    table = wandb.Table(
        columns=["split", "positive_reward_ratio"],
        data=[["train", train_ratio], ["validation", val_ratio]]
    )
    wandb.summary["dataset/positive_reward_ratio"] = table

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.init_from == "base":
        model = cfg.llm.model
    else:
        model = get_best_model(cfg)

    trainer = OfflineRLTrainer(
        model,
        args=SFTConfig(**OmegaConf.to_container(cfg.sft_config, resolve=True)),
        data_collator=OfflineRLDataCollator(tokenizer.pad_token_id),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer
    )
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    evaluate(cfg)


class OfflineRLTrainer(SFTTrainer):
    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        self._signature_columns.append("reward")

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, CausalLMOutputWithPast]:
        _, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch
        )

        # https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer_pt_utils.py#L553
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(-100)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        nll_loss.masked_fill_(padding_mask, 0.0)

        # Offline RL loss
        completion_nll = nll_loss.squeeze(dim=-1).sum(dim=-1)
        loss = (completion_nll * inputs["reward"]).mean()
        return (loss, outputs) if return_outputs else loss


class OfflineRLDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        output["reward"] = torch.tensor([e["reward"] for e in examples])
        return output


if __name__ == "__main__":
    main()
