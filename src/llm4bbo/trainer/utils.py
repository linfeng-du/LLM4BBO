import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


TASK_MAP = {
    "tf8": "TFBind8-Exact-v0",
    "tf10": "TFBind10-Exact-v0",
    "ant": "AntMorphology-Exact-v0",
    "dkitty": "DKittyMorphology-Exact-v0"
}


def update_config(cfg: DictConfig) -> None:
    OmegaConf.update(cfg, "task_name", TASK_MAP[cfg.task], force_add=True)

    config = {
        "llm": cfg.llm,
        "task": cfg.task,
        "stage": cfg.stage,
        "subset_size": cfg.subset_size,
        "evaluate": cfg.evaluate
    }

    if hasattr(cfg, "build_dataset"):
        OmegaConf.update(
            cfg.build_dataset, "task_name", TASK_MAP[cfg.task], force_add=True
        )
        config["build_dataset"] = cfg.build_dataset

    if hasattr(cfg, "init_from"):
        config["init_from"] = cfg.init_from

    if hasattr(cfg, "thinking_budget"):
        config["thinking_budget"] = cfg.thinking_budget

    OmegaConf.update(
        cfg.wandb_init, "config", OmegaConf.create(config), force_add=True
    )


def get_best_model(cfg: DictConfig) -> str:
    assert cfg.init_from in {"sft", "offline_rl"}
    stage_dir = Path(cfg.output_dir).parent.parent / cfg.init_from
    models = []

    for output_dir in stage_dir.iterdir():
        try:
            model = get_model(output_dir)
            results = json.loads((output_dir / "evaluate.json").read_text())
        except (IndexError, FileNotFoundError):
            continue

        models.append((model, results["max_mean"]))

    models.sort(key=lambda x: x[1], reverse=True)
    return models[0][0]


def get_model(output_dir: str) -> str:
    model = list(Path(output_dir).glob("checkpoint-*"))
    return str(model[-1])
