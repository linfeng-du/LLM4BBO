import os

# Prevent `transformers` from using TensorFlow
os.environ["USE_TF"] = "0"

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from trl import SFTConfig, SFTTrainer

from dataset import build_dataset
from resolver import register_resolvers


register_resolvers()


@hydra.main(config_path="../conf", config_name="sft", version_base=None)
def train_sft(cfg: DictConfig):
    wandb.init(project="LLM4BBO", dir="outputs", name=cfg.run_name)

    dataset = build_dataset(stage="sft", **cfg.build_dataset)

    trainer = SFTTrainer(
        cfg.llm.model,
        args=SFTConfig(
            **OmegaConf.to_container(cfg.training_arguments, resolve=True)
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()


if __name__ == "__main__":
    train_sft()
