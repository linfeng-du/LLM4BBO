import os
os.environ["USE_TF"] = "0"

import gc

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch
from trl import SFTConfig, SFTTrainer

from llm4bbo.dataset import build_dataset
from llm4bbo.trainer.hf.evaluate import evaluate
from llm4bbo.trainer.hf.utils import update_config


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    update_config(cfg)
    main_sft(cfg)


def main_sft(cfg: DictConfig) -> None:
    wandb.init(**OmegaConf.to_container(cfg.wandb_init, resolve=True))

    dataset = build_dataset(**cfg.build_dataset)

    trainer = SFTTrainer(
        cfg.llm.model,
        args=SFTConfig(**OmegaConf.to_container(cfg.sft_config, resolve=True)),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    evaluate(cfg)


if __name__ == "__main__":
    main()
