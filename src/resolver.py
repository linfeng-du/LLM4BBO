from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("stringify", _stringify)
    OmegaConf.register_new_resolver("get_model", _get_model)


def _stringify(cfg: DictConfig) -> str:
    args = []
    values = []

    for key, value in cfg.items():
        args.append("".join(word[0] for word in key.split("_")))
        values.append(value)

    return "_".join(f"{a}{v}" for a, v in zip(args, values, strict=True))


def _get_model(run_dir: str) -> str:
    model_dirs = list(Path(run_dir).glob("checkpoint-*"))
    assert len(model_dirs) == 1
    return str(model_dirs[0])
