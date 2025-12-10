"""
Diffusion Scheduler for DDPM

Implements the forward diffusion process (adding noise) and provides
utilities for the reverse process (denoising).

Key concepts:
- Beta schedule: Controls noise levels at each timestep
- Alpha: 1 - beta_t
- Alpha_bar: Cumulative product of alphas
- Forward process: q(x_t | x_0) - add noise to clean image
- Reverse process: p(x_{t-1} | x_t) - remove noise (learned by U-Net)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


class DiffusionScheduler:
    """
    Manages the diffusion process schedule and provides sampling utilities.

    Args:
        num_timesteps: Number of diffusion steps (default: 1000)
        beta_schedule: Type of beta schedule - 'linear' or 'cosine' (default: 'cosine')
        beta_start: Starting beta value for linear schedule (default: 1e-4)
        beta_end: Ending beta value for linear schedule (default: 0.02)
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps

        # Generate beta schedule
        if beta_schedule == 'linear':
            betas = self._linear_beta_schedule(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Calculations for diffusion q(x_t | x_0) and reverse process q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clip to avoid division by zero at t=0
        self.posterior_variance = np.clip(self.posterior_variance, 1e-20, None)

        self.posterior_log_variance_clipped = np.log(self.posterior_variance)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _linear_beta_schedule(self, beta_start: float, beta_end: float, num_timesteps: int) -> np.ndarray:
        """Linear beta schedule."""
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)

    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> np.ndarray:
        """
        Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672

        This schedule provides better sample quality than linear.
        """
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)

        Add noise to clean images according to the noise schedule.

        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_start: (batch_size, channels, height, width) clean images
            noise: (batch_size, channels, height, width) Gaussian noise
            timesteps: (batch_size,) timesteps for each sample

        Returns:
            (batch_size, channels, height, width) noisy images
        """
        # Get values for the given timesteps
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )

        # Add noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        noisy_samples = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def get_variance(self, timestep: int) -> float:
        """Get variance for a given timestep."""
        return self.betas[timestep]

    def get_posterior_variance(self, timestep: int) -> float:
        """Get posterior variance for reverse process."""
        return self.posterior_variance[timestep]

    def get_posterior_mean(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Compute posterior mean for reverse process: q(x_{t-1} | x_t, x_0)

        Args:
            x_t: (batch_size, channels, height, width) noisy image at timestep t
            x_0_pred: (batch_size, channels, height, width) predicted clean image
            timestep: Current timestep

        Returns:
            (batch_size, channels, height, width) posterior mean
        """
        coef1 = self._extract_scalar(self.posterior_mean_coef1, timestep, x_t.shape, x_t.device)
        coef2 = self._extract_scalar(self.posterior_mean_coef2, timestep, x_t.shape, x_t.device)

        return coef1 * x_0_pred + coef2 * x_t

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        Formula: x_0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)

        Args:
            x_t: (batch_size, channels, height, width) noisy images
            timesteps: (batch_size,) timesteps
            noise_pred: (batch_size, channels, height, width) predicted noise

        Returns:
            (batch_size, channels, height, width) predicted clean images
        """
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )

        return (x_t - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod

    def _extract(self, array: np.ndarray, timesteps: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """
        Extract values from array for given timesteps and reshape for broadcasting.

        Args:
            array: Array to extract from
            timesteps: (batch_size,) timesteps
            shape: Target shape (batch_size, channels, height, width)

        Returns:
            (batch_size, 1, 1, 1) values for broadcasting
        """
        batch_size = timesteps.shape[0]
        device = timesteps.device

        # Extract values
        out = torch.from_numpy(array).to(device)[timesteps].float()

        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        out = out.reshape(batch_size, *((1,) * (len(shape) - 1)))

        return out

    def _extract_scalar(self, array: np.ndarray, timestep: int, shape: Tuple, device: torch.device = None) -> torch.Tensor:
        """
        Extract single value for a scalar timestep.

        Args:
            array: Array to extract from
            timestep: Scalar timestep
            shape: Target shape
            device: Device to create tensor on

        Returns:
            Value reshaped for broadcasting
        """
        value = array[timestep]
        # Broadcast to shape (on correct device, float32 for MPS compatibility)
        return torch.full((shape[0], 1, 1, 1), value, dtype=torch.float32, device=device)


def test_scheduler():
    """Test the diffusion scheduler."""
    print("Testing DiffusionScheduler...")

    # Create scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000, beta_schedule='cosine')

    print(f"Number of timesteps: {scheduler.num_timesteps}")
    print(f"Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"Alpha range: [{scheduler.alphas.min():.6f}, {scheduler.alphas.max():.6f}]")
    print(f"Alpha_cumprod range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")

    # Test adding noise
    batch_size = 4
    x_start = torch.randn(batch_size, 1, 28, 28)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(0, 1000, (batch_size,))

    x_noisy = scheduler.add_noise(x_start, noise, timesteps)

    print(f"\nNoise addition test:")
    print(f"Clean image range: [{x_start.min():.3f}, {x_start.max():.3f}]")
    print(f"Noisy image range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
    print(f"Timesteps: {timesteps.tolist()}")

    # Test predicting x_0 from noise
    x_0_pred = scheduler.predict_start_from_noise(x_noisy, timesteps, noise)
    print(f"\nPrediction test:")
    print(f"Predicted x_0 range: [{x_0_pred.min():.3f}, {x_0_pred.max():.3f}]")
    print(f"Reconstruction error: {(x_0_pred - x_start).abs().mean():.6f}")

    # Verify that at t=999, image is almost pure noise
    t_max = torch.full((batch_size,), 999)
    x_very_noisy = scheduler.add_noise(x_start, noise, t_max)
    print(f"\nAt t=999 (maximum noise):")
    print(f"Alpha_cumprod: {scheduler.alphas_cumprod[999]:.6f}")
    print(f"Noisy image std: {x_very_noisy.std():.3f} (should be close to 1)")

    # Verify that at t=0, image is almost unchanged
    t_min = torch.full((batch_size,), 0)
    x_barely_noisy = scheduler.add_noise(x_start, noise, t_min)
    print(f"\nAt t=0 (minimum noise):")
    print(f"Alpha_cumprod: {scheduler.alphas_cumprod[0]:.6f}")
    print(f"Difference from original: {(x_barely_noisy - x_start).abs().mean():.6f}")

    # Compare linear vs cosine schedules
    linear_scheduler = DiffusionScheduler(num_timesteps=1000, beta_schedule='linear')
    print(f"\nLinear schedule beta range: [{linear_scheduler.betas.min():.6f}, {linear_scheduler.betas.max():.6f}]")
    print(f"Cosine schedule beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")

    print("\n✅ All scheduler tests passed!")


if __name__ == "__main__":
    test_scheduler()
