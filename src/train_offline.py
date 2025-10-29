import hydra
import wandb
from omegaconf import DictConfig

from transformers import AutoTokenizer
from trl import SFTConfig

from dataset import build_offline_dataset
from trainer import OfflineRLDataCollator, OfflineRLTrainer


@hydra.main(config_path='../conf', config_name='train_offline', version_base=None)
def train_offline(cfg: DictConfig):
    dataset = build_offline_dataset(**cfg.dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    pad_token_id = (tokenizer.pad_token_id or tokenizer.eos_token_id)
    data_collator = OfflineRLDataCollator(pad_token_id)

    wandb.init(project='LLM4BBO', dir='output', name=cfg.run_name)
    trainer = OfflineRLTrainer(
        cfg.model,
        args=SFTConfig(**cfg.trainer_cfg),
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        processing_class=tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    train_offline()
