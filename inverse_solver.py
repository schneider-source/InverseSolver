from typing import Tuple
import emcee
import numpy as np
import torch
import torch.distributions as dist
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from tqdm import tqdm


class ExampleData:
    """
    Generates synthetic data for a 2D input to 1D output problem.
    The forward model is y = sin(x1) + cos(x2) + noise.
    """
    def __init__(self, num_samples: int = 300, std_noise: float = 0.1):
        self.num_samples = num_samples
        self.std_noise = std_noise
        self.x_obs, self.y_obs = self.make_data()

    def make_data(self):
        """
        Generates synthetic forward data: y = sin(x1) + cos(x2) + noise.
        x values are sampled uniformly from [-3, 3]^2.
        """
        
        x = torch.rand(self.num_samples, 2) * 6 - 3
        y = x[:, 0].sin() + x[:, 1].cos() + self.std_noise * torch.randn(self.num_samples)
        return x, y

    def get_xmask_for_y(self, y_obs: float, tol: float = 0.01, num: int = 500):
        """
        Build an n‐point grid over [-3,3]^2 (sqrt(n) per axis),
        evaluate y_true = sin(x1) + cos(x2), and return a boolean mask for x 
        points where |y_true - y_obs| <= tol.
        """

        x = torch.linspace(-3.0, 3.0, num)
        x1, x2 = torch.meshgrid(x, x, indexing='xy')
        x_flat = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
        y_flat = x_flat[:, 0].sin() + x_flat[:, 1].cos()  # (m*m,)

        return (y_flat - y_obs).abs() <= tol


class GPModel(ExactGP):
    """
    A Gaussian Process model for the forward mapping from x to y.
    This model is trained on observed (x_obs, y_obs) data.
    """
    def __init__(
        self, 
        x_obs: torch.Tensor, 
        y_obs: torch.Tensor, 
        device: torch.device = torch.device("cpu")
    ):
        self.device = device
        likelihood = GaussianLikelihood().to(device)
        super().__init__(x_obs.to(device), y_obs.to(device), likelihood)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.to(device)
        
        # Define a uniform prior distribution for x: U([-3, 3]^2)
        self.log_prior_x = dist.Independent(dist.Uniform(
            -torch.full((x_obs.size(-1),), 3., device=device), 
            torch.full((x_obs.size(-1),), 3., device=device), 
            validate_args=False
        ), 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

    def train_forward_gp(self, lr: float = 0.01, epochs: int = 1000):
        """
        Trains the Gaussian Process model by optimizing its hyperparameters
        to maximize the marginal log likelihood.
        """
        
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        with tqdm(range(1000), desc="GP training") as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                loss = -mll(self(self.train_inputs[0]), self.train_targets)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

        self.eval()
        self.likelihood.eval()
        
        # ensure caches are populated after training
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.detach_test_caches(state=False):
                _ = self.likelihood(self(self.train_inputs[0]))
                
    def log_posterior_x(self, x: torch.Tensor, y: float) -> torch.Tensor:
        """
        Computes the log of the unnormalized posterior probability of x given an observed y.
        This is used by the MCMC sampler.
        The log posterior is calculated as: log p(x|y) ∝ log p(y|x) + log p(x)
        where:
        - p(y|x) is the likelihood of observing y_obs given x, obtained from the GP.
        - p(x) is the prior probability of x.
        
        x: shape (n_walkers, x_dim)
        y: observed scalar value y_obs
        returns: log p(x | y) = log p(y | x) + log p(x); shape (n_walkers,)
        """

        x = x.to(self.device).float()
        
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.detach_test_caches(state=False):
                 
                # Log prior probability of x
                log_prior = self.log_prior_x.log_prob(x)
                
                # Get GP's predictive distribution for y given x
                post = self.likelihood(self(x))
                mu  = post.mean
                var = post.variance.clamp_min(1e-6)
                
                # Log likelihood of observing y given GP's prediction (Gaussian PDF)
                log_likelihood = - ((y - mu) ** 2 / var - torch.log(2 * torch.pi * var)) / 2
                log_post = log_prior + log_likelihood
        
        return log_post

    def sample_posterior_x(self, y: float, num_walkers: int = 50, steps: int = 1000, burn: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw MCMC samples from p(x|y) using emcee with vectorized proposals.
        
        y: The observed y value for which to sample p(x|y).
        Returns: Tuple of (flat_samples, flat_log_probabilities)
        """
        p0 = self.log_prior_x.sample((num_walkers,))
        
        sampler = emcee.EnsembleSampler(
            nwalkers=num_walkers, 
            ndim=p0.size(-1), 
            log_prob_fn=lambda x: self.log_posterior_x(torch.from_numpy(x), y).cpu().numpy(),
            vectorize=True,
        )
        sampler.run_mcmc(p0.cpu().numpy(), steps, progress=True)
        return sampler.get_chain(discard=burn, flat=True), sampler.get_log_prob(discard=burn, flat=True)
        
        
def main():
    """
    Main script to demonstrate inverse problem solving using GP-MCMC.
    1. Generates synthetic data.
    2. Trains a GP on this data to model the forward process y = f(x).
    3. For given y_obs values, uses MCMC to sample from the posterior p(x|y_obs).
    4. Plots the predicted posterior density p(x|y_obs) and the true x region.
    """
    
    import matplotlib.pyplot as plt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Generate synthetic data
    data = ExampleData(num_samples=500, std_noise=0.05)
    x_obs, y_obs = data.make_data()

    # 2. Train GP model on the forward data
    model = GPModel(x_obs=x_obs, y_obs=y_obs, device=device)
    model.train_forward_gp(lr=0.01, epochs=1000)

    # Define target y values for which to perform inverse inference
    y_vals = [-.5, 0., .5]

    res_pred = 100
    num_grid = 500
    tol = 0.01

    fig, axes = plt.subplots(len(y_vals), 1, figsize=(6, 4 * len(y_vals)))

    for ax, y0 in zip(axes, y_vals):
        # 3. Sample from posterior p(x|y0) using MCMC
        samples, log_prob = model.sample_posterior_x(y0, num_walkers=200, steps=1000, burn=200)
        hist_pred, x_edges, y_edges = np.histogram2d(
            samples[:, 0], samples[:, 1],
            bins=res_pred, range=[[-3, 3], [-3, 3]], density=True
        )
        im = ax.imshow(hist_pred.T, origin='lower', extent=[-3,3,-3,3], cmap='viridis')
        fig.colorbar(im, ax=ax, label='Predicted p(x|y)')

        # Overlay the true region of x that corresponds to y0
        mask = data.get_xmask_for_y(y0, tol, num_grid).reshape(num_grid, num_grid).cpu().numpy()
        rgba = np.zeros((num_grid, num_grid, 4))
        rgba[mask] = [1.0, 1.0, 1.0, 0.8]
        ax.imshow(rgba, origin='lower', extent=[-3,3,-3,3])

        ax.set_title(f"GP-MCMC p(x|y={y0:.2f})")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend([plt.Rectangle((0,0), 1, 1, fc=rgba[mask][0])], ['ground truth'])

    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()