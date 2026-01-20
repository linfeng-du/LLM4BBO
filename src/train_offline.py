import os
from typing import Any

# Prevent Transformers from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from peft import LoraConfig
from transformers import AutoTokenizer, pipeline
from transformers.modeling_outputs import CausalLMOutputWithPast

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from dataset import build_rl_dataset
from test import test


@hydra.main(
    config_path="../conf",
    config_name="train_offline",
    version_base=None
)
def train_offline(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dataset = build_rl_dataset(mode="offline", **cfg.dataset)

    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    trainer = OfflineRLTrainer(
        cfg.llm.model,
        args=SFTConfig(**cfg.trainer_args),
        data_collator=OfflineRLDataCollator(pad_token_id),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(**OmegaConf.to_container(cfg.llm.lora_config))
    )
    trainer.train()

    pipe = pipeline(
        task="text-generation",
        model=trainer.model,
        tokenizer=trainer.processing_class,
        **cfg.llm.hf_generation_kwargs
    )
    test(pipe, cfg)


class OfflineRLDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        output["reward"] = torch.tensor([e["reward"] for e in examples])
        return output


class OfflineRLTrainer(SFTTrainer):
    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        self._signature_columns.append("reward")

    def compute_loss(
        self,
        model: nn.Module,
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

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["labels"][:, 1:].contiguous()

        logps = logits.log_softmax(dim=-1)
        token_logps = (
            logps.gather(dim=-1, index=labels.unsqueeze(dim=-1).clamp(min=0))
            .squeeze(dim=-1)
        )
        token_logps.masked_fill_(labels == -100, value=0.0)

        completion_logp = token_logps.sum(dim=-1)
        loss = -(completion_logp * inputs["reward"]).mean()
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    train_offline()
