"""
Improved Gaussian Process implementations with better hyperparameter initialization.
Includes both GPyTorch and scikit-learn implementations.
"""

import torch
import numpy as np
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


class ImprovedExactGPModel(gpytorch.models.ExactGP):
    """
    Improved GPyTorch model with better hyperparameter initialization.
    """
    def __init__(self, train_x, train_y, likelihood, rbf_only=False):
        """
        Args:
            train_x: Training inputs
            train_y: Training targets
            likelihood: GPyTorch likelihood
            rbf_only: If True, use RBF kernel without ScaleKernel (simpler, matches scikit-learn RBF-only)
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF Kernel with better initialization
        # Initialize lengthscale based on data spread
        lengthscale_init = self._estimate_lengthscale(train_x)
        outputscale_init = self._estimate_outputscale(train_y)
        
        if rbf_only:
            # RBF kernel only (no ScaleKernel) - simpler, matches scikit-learn RBF-only
            self.covar_module = gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Positive()
            )
            # For RBF-only, we still initialize lengthscale
            lengthscale_val = lengthscale_init.item() if isinstance(lengthscale_init, torch.Tensor) else float(lengthscale_init)
            self.covar_module.initialize(lengthscale=lengthscale_val)
        else:
            # RBF kernel with ScaleKernel (default, more flexible)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_constraint=gpytorch.constraints.Positive()
                )
            )
            # Initialize hyperparameters using GPyTorch's initialize method
            # Convert to Python float to avoid shape issues
            lengthscale_val = lengthscale_init.item() if isinstance(lengthscale_init, torch.Tensor) else float(lengthscale_init)
            outputscale_val = outputscale_init.item() if isinstance(outputscale_init, torch.Tensor) else float(outputscale_init)
            
            self.covar_module.base_kernel.initialize(lengthscale=lengthscale_val)
            self.covar_module.initialize(outputscale=outputscale_val)
        
        # Initialize likelihood noise based on data variance
        noise_init = self._estimate_noise(train_y)
        noise_val = noise_init.item() if isinstance(noise_init, torch.Tensor) else float(noise_init)
        self.likelihood.initialize(noise=noise_val)
        
        self.rbf_only = rbf_only
    
    def _estimate_lengthscale(self, x):
        """Estimate initial lengthscale from data.
        
        Returns a scalar value for single lengthscale (not ARD).
        GPyTorch RBF kernel expects shape [1, 1] for single lengthscale.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Use median pairwise distance as initial lengthscale
        # Sample a subset for efficiency
        n_samples = min(100, x.shape[0])
        if n_samples < 2:
            # Fallback: use mean of standard deviations across dimensions
            std_per_dim = torch.std(x, dim=0)
            lengthscale = torch.mean(std_per_dim).clamp(min=0.1, max=10.0)
            # Ensure it's a scalar tensor
            if lengthscale.dim() == 0:
                lengthscale = lengthscale.unsqueeze(0)
            return lengthscale
        
        indices = torch.randperm(x.shape[0])[:n_samples]
        x_sample = x[indices]
        
        # Compute pairwise distances
        dists = torch.cdist(x_sample, x_sample)
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        dists = dists[mask]
        
        # Filter out zero distances
        dists = dists[dists > 1e-6]
        
        if len(dists) == 0:
            # Fallback: use mean of standard deviations across dimensions
            std_per_dim = torch.std(x, dim=0)
            lengthscale = torch.mean(std_per_dim).clamp(min=0.1, max=10.0)
            # Ensure it's a scalar tensor
            if lengthscale.dim() == 0:
                lengthscale = lengthscale.unsqueeze(0)
            return lengthscale
        
        # Use median distance as lengthscale estimate
        lengthscale = torch.median(dists)
        
        # Ensure it's reasonable (not too small or too large)
        lengthscale = torch.clamp(lengthscale, min=0.1, max=10.0)
        
        # Ensure it's a scalar tensor (not 0-d)
        if lengthscale.dim() == 0:
            lengthscale = lengthscale.unsqueeze(0)
        
        # Return scalar tensor (GPyTorch will handle the shape [1, 1] internally)
        return lengthscale
    
    def _estimate_outputscale(self, y):
        """Estimate initial outputscale from data variance."""
        y_var = torch.var(y)
        # Outputscale should be similar to data variance
        outputscale = torch.clamp(y_var, min=0.1, max=10.0)
        return outputscale
    
    def _estimate_noise(self, y):
        """Estimate initial noise from data."""
        # Use a fraction of data variance as noise
        y_var = torch.var(y)
        noise = torch.clamp(y_var * 0.1, min=1e-4, max=1.0)
        return noise
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_lengthscale(self):
        """Get the lengthscale parameter."""
        if self.rbf_only:
            return self.covar_module.lengthscale
        else:
            return self.covar_module.base_kernel.lengthscale
    
    def get_outputscale(self):
        """Get the outputscale parameter (None if rbf_only=True)."""
        if self.rbf_only:
            return None
        else:
            return self.covar_module.outputscale


class SklearnGPModel:
    """
    Wrapper for scikit-learn's GaussianProcessRegressor.
    Uses L-BFGS-B optimization (numerical method) instead of gradient descent.
    
    Note: scikit-learn's GaussianProcessRegressor uses L-BFGS-B optimizer,
    which is a quasi-Newton method. It doesn't support gradient descent directly,
    but L-BFGS-B is generally more robust than gradient descent for GP optimization.
    """
    def __init__(self, alpha=1e-6, n_restarts_optimizer=10, rbf_only=False):
        """
        Args:
            alpha: Value added to the diagonal of the kernel matrix during fitting.
                  Can be interpreted as the variance of additional Gaussian measurement noise.
            n_restarts_optimizer: Number of restarts of the optimizer for finding the kernel's parameters.
            rbf_only: If True, use only RBF kernel. If False, use ConstantKernel * RBF + WhiteKernel.
        """
        if rbf_only:
            # RBF kernel only (simpler, matches GPyTorch default)
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        else:
            # RBF kernel with automatic relevance determination (ARD)
            # ConstantKernel * RBF + WhiteKernel for noise
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,  # Normalize targets to zero mean and unit variance
            random_state=42
        )
        self.is_fitted = False
        self.rbf_only = rbf_only
    
    def fit(self, X, y):
        """Fit the GP model."""
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.squeeze()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X, return_std=True):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if return_std:
            mean, std = self.model.predict(X, return_std=True)
            return mean, std
        else:
            mean = self.model.predict(X, return_std=False)
            return mean
    
    def get_kernel_params(self):
        """Get optimized kernel parameters."""
        if not self.is_fitted:
            return None
        return self.model.kernel_.get_params()

