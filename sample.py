"""
Sampling script for generating images with trained diffusion model

Generates images from trained model and saves them to disk.
Supports:
- Conditional generation (by class)
- Unconditional generation
- Classifier-free guidance
- Progressive denoising visualization
"""

import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.ddpm import DDPMSampler
from utils import (
    EMA, get_device, load_checkpoint, denormalize_from_range
)


def generate_samples(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    device: torch.device,
    config: Config,
    class_labels: torch.Tensor = None,
    guidance_scale: float = 3.0,
    num_samples: int = 16,
    save_path: str = 'samples.png',
    show_progress: bool = True,
):
    """
    Generate samples from trained model.

    Args:
        model: Trained U-Net model
        scheduler: Diffusion scheduler
        device: Device to use
        config: Configuration
        class_labels: (num_samples,) class labels for conditional generation
        guidance_scale: Classifier-free guidance scale
        num_samples: Number of samples to generate
        save_path: Path to save generated images
        show_progress: Whether to show progress bar
    """
    sampler = DDPMSampler(scheduler, model, device)

    print(f"\nGenerating {num_samples} samples...")
    print(f"Guidance scale: {guidance_scale}")

    # Generate samples
    samples = sampler.sample(
        batch_size=num_samples,
        channels=config.model.in_channels,
        height=config.data.image_size,
        width=config.data.image_size,
        class_labels=class_labels,
        guidance_scale=guidance_scale,
        progress_bar=show_progress,
    )

    # Denormalize to [0, 1]
    samples = denormalize_from_range(samples, config.data.normalize_to)
    samples = torch.clamp(samples, 0, 1)

    # Create grid and save
    nrow = int(num_samples ** 0.5)
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)

    save_image(grid, save_path)
    print(f"Samples saved to {save_path}")

    return samples


def generate_class_grid(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    device: torch.device,
    config: Config,
    samples_per_class: int = 4,
    guidance_scale: float = 3.0,
    save_path: str = 'class_grid.png',
):
    """
    Generate a grid of samples with one row per class.

    Args:
        model: Trained model
        scheduler: Diffusion scheduler
        device: Device
        config: Configuration
        samples_per_class: Number of samples per class
        guidance_scale: Guidance scale
        save_path: Save path
    """
    num_classes = config.model.num_classes
    all_samples = []

    print(f"\nGenerating {samples_per_class} samples per class...")

    for class_idx in range(num_classes):
        print(f"Generating class {class_idx}...")

        class_labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

        samples = generate_samples(
            model, scheduler, device, config,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            num_samples=samples_per_class,
            save_path=f'/tmp/class_{class_idx}.png',
            show_progress=False,
        )

        all_samples.append(samples)

    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)

    # Create grid
    grid = make_grid(all_samples, nrow=samples_per_class, padding=2, normalize=False)
    save_image(grid, save_path)
    print(f"\nClass grid saved to {save_path}")

    return all_samples


def generate_progressive_denoising(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    device: torch.device,
    config: Config,
    class_label: int = 0,
    guidance_scale: float = 3.0,
    save_dir: str = 'progressive',
):
    """
    Generate and visualize progressive denoising process.

    Args:
        model: Trained model
        scheduler: Diffusion scheduler
        device: Device
        config: Configuration
        class_label: Class to generate
        guidance_scale: Guidance scale
        save_dir: Directory to save progressive images
    """
    os.makedirs(save_dir, exist_ok=True)

    sampler = DDPMSampler(scheduler, model, device)

    print(f"\nGenerating progressive denoising for class {class_label}...")

    # Initial noise
    x_start = torch.randn(1, config.model.in_channels, config.data.image_size, config.data.image_size, device=device)

    # Class label
    class_labels = torch.tensor([class_label], device=device)

    # Timesteps to save
    timesteps_to_save = config.sampling.progressive_timesteps

    # Generate with progressive saving
    progressive = sampler.progressive_denoising(
        x_start,
        timesteps_to_save=timesteps_to_save,
        class_labels=class_labels,
        guidance_scale=guidance_scale,
    )

    # Save each step
    saved_images = []
    for t, x_t, x_0_pred in progressive:
        # Denormalize
        x_t_vis = denormalize_from_range(x_t, config.data.normalize_to)
        x_0_pred_vis = denormalize_from_range(x_0_pred, config.data.normalize_to)

        x_t_vis = torch.clamp(x_t_vis, 0, 1)
        x_0_pred_vis = torch.clamp(x_0_pred_vis, 0, 1)

        # Save
        save_image(x_t_vis, os.path.join(save_dir, f't_{t:04d}_noisy.png'))
        save_image(x_0_pred_vis, os.path.join(save_dir, f't_{t:04d}_predicted.png'))

        saved_images.append((t, x_t_vis, x_0_pred_vis))

        print(f"Saved t={t}")

    # Create comparison grid
    all_x_t = torch.cat([x[1] for x in saved_images], dim=0)
    all_x_0 = torch.cat([x[2] for x in saved_images], dim=0)

    grid_x_t = make_grid(all_x_t, nrow=len(saved_images), padding=2)
    grid_x_0 = make_grid(all_x_0, nrow=len(saved_images), padding=2)

    save_image(grid_x_t, os.path.join(save_dir, 'progressive_noisy.png'))
    save_image(grid_x_0, os.path.join(save_dir, 'progressive_predicted.png'))

    print(f"\nProgressive denoising saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Classifier-free guidance scale')
    parser.add_argument('--class_label', type=int, default=None, help='Class label for conditional generation')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'class_grid', 'progressive'],
                        help='Generation mode')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA weights')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    print(f"Loading config from {args.config}...")
    config = Config.load(args.config)

    # Get device
    device = get_device(args.device)

    # Create model
    print("Creating model...")
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        channel_mults=config.model.channel_mults,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=0.0,  # No dropout during inference
        num_classes=config.model.num_classes,
        embed_dim=config.model.embed_dim,
    ).to(device)

    # Create EMA if needed
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=config.training.ema_decay)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, ema=ema, path=args.checkpoint, device=device)

    # Apply EMA weights if requested
    if ema is not None:
        print("Using EMA weights for generation")
        ema.apply_shadow()

    # Create scheduler
    scheduler = DiffusionScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )

    model.eval()

    # Generate based on mode
    if args.mode == 'grid':
        # Simple grid of samples
        if args.class_label is not None:
            class_labels = torch.full((args.num_samples,), args.class_label, device=device, dtype=torch.long)
        else:
            class_labels = None

        save_path = os.path.join(args.output_dir, f'samples_guidance{args.guidance_scale:.1f}.png')
        generate_samples(
            model, scheduler, device, config,
            class_labels=class_labels,
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
            save_path=save_path,
        )

    elif args.mode == 'class_grid':
        # Grid with one row per class
        samples_per_class = max(1, args.num_samples // config.model.num_classes)
        save_path = os.path.join(args.output_dir, f'class_grid_guidance{args.guidance_scale:.1f}.png')
        generate_class_grid(
            model, scheduler, device, config,
            samples_per_class=samples_per_class,
            guidance_scale=args.guidance_scale,
            save_path=save_path,
        )

    elif args.mode == 'progressive':
        # Progressive denoising visualization
        class_label = args.class_label if args.class_label is not None else 0
        save_dir = os.path.join(args.output_dir, f'progressive_class{class_label}')
        generate_progressive_denoising(
            model, scheduler, device, config,
            class_label=class_label,
            guidance_scale=args.guidance_scale,
            save_dir=save_dir,
        )

    print("\n✅ Generation complete!")

    # Restore original weights if using EMA
    if ema is not None:
        ema.restore()


if __name__ == "__main__":
    import torch.nn as nn  # Import needed for generate_samples
    main()
