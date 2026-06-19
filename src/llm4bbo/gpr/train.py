import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from llm4bbo.dataset import evenly_spaced_indices, load_task_data


def train(task_name: str, num_designs: int, seed: int = 42):
    task, x, y, _ = load_task_data(task_name)

    if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
        x = task.to_logits(x).reshape(len(x), -1)

    train_index = evenly_spaced_indices(y, num_designs)

    rng = np.random.default_rng(seed)
    remain_index = np.setdiff1d(np.arange(len(x)), train_index)
    val_index = rng.choice(remain_index, num_designs, replace=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.from_numpy(x[train_index]).to(device, torch.float64)
    y_train = torch.from_numpy(y[train_index]).to(device, torch.float64)
    x_val = torch.from_numpy(x[val_index]).to(device, torch.float64)

    model = SingleTaskGP(x_train, y_train, input_transform=Normalize(x_train.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()

    with torch.no_grad():
        y_pred = model.posterior(x_val).mean.cpu().numpy()

    results = {
        "r2": r2_score(y[val_index], y_pred),
        "spearman": spearmanr(y[val_index], y_pred)[0],
    }
    print(results)


num_designs = 500
train("TFBind8-Exact-v0", num_designs)
train("TFBind10-Exact-v0", num_designs)
train("AntMorphology-Exact-v0", num_designs)
train("DKittyMorphology-Exact-v0", num_designs)
