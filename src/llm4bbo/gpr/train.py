from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from llm4bbo.dataset import evenly_spaced_indices, load_task_data


def estimate_lengthscale(x: torch.Tensor) -> float:
    """Median pairwise distance on a random subsample."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    n_samples = min(128, x.shape[0])
    if n_samples < 2:
        return 1.0

    indices = torch.randperm(x.shape[0], device=x.device)[:n_samples]
    x_sample = x[indices]
    dists = torch.cdist(x_sample, x_sample)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    dists = dists[mask]
    dists = dists[dists > 1e-8]

    if dists.numel() == 0:
        return 1.0

    return float(torch.clamp(torch.median(dists), min=1e-2, max=1e2).item())


def estimate_outputscale(y: torch.Tensor) -> float:
    y_var = torch.var(y)
    return float(torch.clamp(y_var, min=1e-3, max=1e3).item())


def estimate_noise(y: torch.Tensor) -> float:
    y_var = torch.var(y)
    return float(torch.clamp(y_var * 0.05, min=1e-4, max=1.0).item())


def _initialize_hyperparameters(
    model: SingleTaskGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    use_outputscale: bool,
) -> None:
    lengthscale_init = estimate_lengthscale(train_x)
    noise_init = estimate_noise(train_y.reshape(-1))

    if use_outputscale:
        outputscale_init = estimate_outputscale(train_y.reshape(-1))
        model.covar_module.base_kernel.initialize(lengthscale=lengthscale_init)
        model.covar_module.initialize(outputscale=outputscale_init)
    else:
        model.covar_module.initialize(lengthscale=lengthscale_init)

    model.likelihood.initialize(noise=noise_init)


def _get_lengthscale(model: SingleTaskGP) -> np.ndarray:
    covar = model.covar_module
    if isinstance(covar, ScaleKernel):
        return covar.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    return covar.lengthscale.detach().cpu().numpy().ravel()


def _uses_outputscale(model: SingleTaskGP) -> bool:
    return isinstance(model.covar_module, ScaleKernel)


def fit_model(
    model: SingleTaskGP,
    *,
    lr: float = 0.05,
    n_epochs: int = 300,
    verbose: bool = True,
) -> SingleTaskGP:
    """Adam training loop for a SingleTaskGP (same as gp_kernel/gp_ye_gpt.py)."""
    model.train()
    model.likelihood.train()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_x = model.train_inputs[0]
    train_y = model.train_targets

    epoch_iter = range(1, n_epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="GP training", leave=False)

    for epoch in epoch_iter:
        opt.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        opt.step()

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == n_epochs):
            ls = _get_lengthscale(model)
            noise = float(model.likelihood.noise.detach().cpu().item())
            postfix = {
                "loss": f"{loss.item():.4f}",
                "noise": f"{noise:.6f}",
                "ls_med": f"{np.median(ls):.4f}",
            }
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(postfix)

    model.eval()
    model.likelihood.eval()

    if verbose:
        ls = _get_lengthscale(model)
        noise = float(model.likelihood.noise.detach().cpu().item())
        print(
            f"[GP] Fitted lengthscale(s): n={len(ls)}, median={np.median(ls):.4f}, "
            f"min={ls.min():.4f}, max={ls.max():.4f}"
        )
        print(f"[GP] Fitted noise: {noise:.6f}")
        if _uses_outputscale(model):
            outputscale = float(model.covar_module.outputscale.detach().cpu().item())
            print(f"[GP] Fitted outputscale: {outputscale:.6f}")

    return model


def load_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    use_ard: bool = False,
    use_outputscale: bool = True,
    noise_floor: float = 1e-4,
) -> SingleTaskGP:
    """
    Build a SingleTaskGP with standard RBF kernel and data-driven hyperparameter
    initialization (same strategy as gp_kernel/gp_ye_gpt.py ExactRBFGPModel).
    """
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    dtype = train_x.dtype
    device = train_x.device
    d = train_x.shape[-1]

    ard_num_dims = d if use_ard else None
    base_kernel = RBFKernel(
        ard_num_dims=ard_num_dims,
        lengthscale_constraint=GreaterThan(0.0),
    )
    covar_module = ScaleKernel(base_kernel) if use_outputscale else base_kernel

    likelihood = GaussianLikelihood(
        noise_constraint=GreaterThan(noise_floor),
    ).to(device=device, dtype=dtype)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=None,
    ).to(device=device, dtype=dtype)

    _initialize_hyperparameters(
        model,
        train_x,
        train_y,
        use_outputscale=use_outputscale,
    )
    return model


def train(task_name: str, num_designs: int, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    task, x, y, _ = load_task_data(task_name)

    if task_name in {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}:
        x = task.to_logits(x).reshape(len(x), -1)

    train_index = evenly_spaced_indices(y, num_designs)
    remain_index = np.setdiff1d(np.arange(len(x)), train_index)
    val_index = np.random.choice(remain_index, num_designs, replace=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.from_numpy(x[train_index]).to(device, torch.float64)
    y_train = torch.from_numpy(y[train_index]).to(device, torch.float64)
    x_val = torch.from_numpy(x[val_index]).to(device, torch.float64)

    model = fit_model(load_model(x_train, y_train))

    with torch.no_grad():
        y_pred = model.posterior(x_val).mean.cpu().numpy()

    results = {
        "r2": r2_score(y[val_index], y_pred),
        "spearman": spearmanr(y[val_index], y_pred)[0].item(),
    }
    print(results)


num_designs = 500
train("TFBind8-Exact-v0", num_designs)
train("TFBind10-Exact-v0", num_designs)
train("AntMorphology-Exact-v0", num_designs)
train("DKittyMorphology-Exact-v0", num_designs)
