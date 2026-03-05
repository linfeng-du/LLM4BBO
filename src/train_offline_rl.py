import os
from typing import Any

# Prevent `transformers` from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from dataset import build_dataset
from resolver import register_resolvers


register_resolvers()


@hydra.main(config_path="../conf", config_name="offline_rl", version_base=None)
def train_offline_rl(cfg: DictConfig):
    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    dataset = build_dataset(stage="offline_rl", **cfg.build_dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainer = OfflineRLTrainer(
        cfg.base_model,
        args=SFTConfig(
            **OmegaConf.to_container(cfg.training_arguments, resolve=True)
        ),
        data_collator=OfflineRLDataCollator(tokenizer.pad_token_id),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer
    )
    trainer.train()


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
    train_offline_rl()
