"""
DDPM Sampler - Denoising Diffusion Probabilistic Models

Implements the reverse diffusion process (denoising) to generate images from noise.

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
https://arxiv.org/abs/2006.11239
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm

from .scheduler import DiffusionScheduler


class DDPMSampler:
    """
    DDPM sampling for generating images from noise.

    Iteratively denoises pure noise to generate images.
    Supports classifier-free guidance for conditional generation.

    Args:
        scheduler: DiffusionScheduler instance
        model: Trained U-Net model
        device: Device to run sampling on
    """
    def __init__(
        self,
        scheduler: DiffusionScheduler,
        model: nn.Module,
        device: torch.device
    ):
        self.scheduler = scheduler
        self.model = model
        self.device = device
        self.num_timesteps = scheduler.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        Generate images using DDPM sampling.

        Args:
            batch_size: Number of images to generate
            channels: Number of channels (1 for grayscale, 3 for RGB)
            height: Image height
            width: Image width
            class_labels: (batch_size,) class labels for conditional generation
            guidance_scale: Strength of classifier-free guidance (1.0 = no guidance)
            progress_bar: Whether to show progress bar

        Returns:
            (batch_size, channels, height, width) generated images
        """
        self.model.eval()

        # Start from pure noise
        x = torch.randn(batch_size, channels, height, width, device=self.device)

        # Iteratively denoise
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress_bar:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            # Current timestep tensor
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            if class_labels is not None and guidance_scale != 1.0:
                # Classifier-free guidance: interpolate between conditional and unconditional
                noise_pred = self._predict_noise_with_guidance(
                    x, t_tensor, class_labels, guidance_scale
                )
            else:
                # Standard prediction
                noise_pred = self.model(x, t_tensor, class_labels)

            # Compute predicted x_0
            x_0_pred = self.scheduler.predict_start_from_noise(x, t_tensor, noise_pred)

            # Clip predicted x_0 to valid range [-1, 1]
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Compute previous sample x_{t-1}
            if t > 0:
                # Add noise for non-final steps
                x = self._reverse_step(x, x_0_pred, t)
            else:
                # Final step: no noise
                x = x_0_pred

        return x

    def _predict_noise_with_guidance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Predict noise with classifier-free guidance.

        Interpolates between conditional and unconditional predictions:
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        Args:
            x: (batch_size, channels, height, width) noisy images
            t: (batch_size,) timesteps
            class_labels: (batch_size,) class labels
            guidance_scale: Guidance strength

        Returns:
            (batch_size, channels, height, width) guided noise prediction
        """
        batch_size = x.shape[0]

        # Conditional prediction
        noise_cond = self.model(x, t, class_labels, force_drop_class=False)

        # Unconditional prediction
        noise_uncond = self.model(x, t, class_labels, force_drop_class=True)

        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_pred

    def _reverse_step(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Compute x_{t-1} from x_t and predicted x_0.

        Uses the posterior mean and adds noise scaled by posterior variance.

        Args:
            x_t: (batch_size, channels, height, width) current noisy image
            x_0_pred: (batch_size, channels, height, width) predicted clean image
            t: Current timestep

        Returns:
            (batch_size, channels, height, width) x_{t-1}
        """
        # Compute posterior mean
        mean = self.scheduler.get_posterior_mean(x_t, x_0_pred, t)

        # Get posterior variance
        variance = self.scheduler.get_posterior_variance(t)

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t) if t > 0 else 0.0

        # x_{t-1} = mean + sqrt(variance) * noise
        # Create variance tensor on same device as x_t (float32 for MPS compatibility)
        x_prev = mean + torch.sqrt(torch.tensor(variance, dtype=torch.float32, device=x_t.device)) * noise

        return x_prev

    @torch.no_grad()
    def progressive_denoising(
        self,
        x_start: torch.Tensor,
        timesteps_to_save: list = None,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> list:
        """
        Generate images and save intermediate steps for visualization.

        Useful for understanding the denoising process.

        Args:
            x_start: (batch_size, channels, height, width) starting noise
            timesteps_to_save: List of timesteps to save (e.g., [999, 750, 500, 250, 0])
            class_labels: (batch_size,) class labels
            guidance_scale: Guidance strength

        Returns:
            List of (timestep, image) tuples
        """
        if timesteps_to_save is None:
            # Save 10 evenly spaced timesteps
            timesteps_to_save = list(range(self.num_timesteps - 1, -1, -self.num_timesteps // 10))

        self.model.eval()
        x = x_start.clone()
        saved_images = []

        for t in range(self.num_timesteps - 1, -1, -1):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            if class_labels is not None and guidance_scale != 1.0:
                noise_pred = self._predict_noise_with_guidance(
                    x, t_tensor, class_labels, guidance_scale
                )
            else:
                noise_pred = self.model(x, t_tensor, class_labels)

            # Compute predicted x_0
            x_0_pred = self.scheduler.predict_start_from_noise(x, t_tensor, noise_pred)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Save if requested
            if t in timesteps_to_save:
                saved_images.append((t, x.clone(), x_0_pred.clone()))

            # Next step
            if t > 0:
                x = self._reverse_step(x, x_0_pred, t)
            else:
                x = x_0_pred

        return saved_images


def test_sampler():
    """Test DDPM sampler (without actual training)."""
    print("Testing DDPMSampler...")

    from ..models.unet import UNet

    # Create small model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[],
        dropout=0.0,
        num_classes=10,
        embed_dim=128,
    ).to(device)

    # Create scheduler
    scheduler = DiffusionScheduler(num_timesteps=100, beta_schedule='cosine')

    # Create sampler
    sampler = DDPMSampler(scheduler, model, device)

    # Test sampling (unconditional)
    print("\nTesting unconditional sampling...")
    samples_uncond = sampler.sample(
        batch_size=2,
        channels=1,
        height=28,
        width=28,
        class_labels=None,
        guidance_scale=1.0,
        progress_bar=False,
    )
    print(f"Unconditional samples shape: {samples_uncond.shape}")
    print(f"Sample range: [{samples_uncond.min():.3f}, {samples_uncond.max():.3f}]")

    # Test conditional sampling
    print("\nTesting conditional sampling...")
    class_labels = torch.tensor([0, 5], device=device)
    samples_cond = sampler.sample(
        batch_size=2,
        channels=1,
        height=28,
        width=28,
        class_labels=class_labels,
        guidance_scale=2.0,
        progress_bar=False,
    )
    print(f"Conditional samples shape: {samples_cond.shape}")
    print(f"Class labels: {class_labels.tolist()}")

    # Test progressive denoising
    print("\nTesting progressive denoising...")
    x_start = torch.randn(1, 1, 28, 28, device=device)
    timesteps_to_save = [99, 75, 50, 25, 0]
    progressive = sampler.progressive_denoising(
        x_start,
        timesteps_to_save=timesteps_to_save,
        class_labels=torch.tensor([3], device=device),
        guidance_scale=1.0,
    )
    print(f"Saved {len(progressive)} intermediate steps")
    for t, x_t, x_0_pred in progressive:
        print(f"  t={t}: x_t range [{x_t.min():.2f}, {x_t.max():.2f}], "
              f"x_0_pred range [{x_0_pred.min():.2f}, {x_0_pred.max():.2f}]")

    print("\n✅ All sampler tests passed!")


if __name__ == "__main__":
    test_sampler()
