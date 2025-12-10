"""
Training script for Diffusion Model

Trains a U-Net to predict noise in the diffusion process.
Includes:
- Fashion-MNIST data loading
- Training loop with EMA
- WandB logging
- Sample generation for visualization
- Checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import wandb
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import Config, get_baseline_config, get_small_config
from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.ddpm import DDPMSampler
from utils import (
    EMA, set_seed, get_device, count_parameters,
    normalize_to_range, denormalize_from_range,
    save_checkpoint, load_checkpoint, create_output_dirs, get_lr
)


class NormalizeTransform:
    """Normalize transform that can be pickled (for multiprocessing)."""
    def __init__(self, target_range):
        self.target_range = target_range

    def __call__(self, x):
        return normalize_to_range(x, self.target_range)


def get_dataloaders(config: Config):
    """
    Create Fashion-MNIST dataloaders.

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader
    """
    # Transform: ToTensor() gives [0, 1], we normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        NormalizeTransform(config.data.normalize_to),  # Pickle-able version
    ])

    # Load Fashion-MNIST
    if config.data.dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(
            root=config.data.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        val_dataset = datasets.FashionMNIST(
            root=config.data.data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif config.data.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            root=config.data.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        val_dataset = datasets.MNIST(
            root=config.data.data_dir,
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    print(f"\nDataset: {config.data.dataset}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
    epoch: int,
    config: Config,
    use_wandb: bool = True,
):
    """
    Train for one epoch.

    Args:
        model: U-Net model
        scheduler: Diffusion scheduler
        train_loader: Training dataloader
        optimizer: Optimizer
        ema: EMA object
        device: Device
        epoch: Current epoch
        config: Configuration
        use_wandb: Whether to log to WandB
    """
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Sample random timesteps
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, scheduler.num_timesteps, (batch_size,), device=device
        ).long()

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise to images (forward diffusion)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Classifier-free guidance: randomly drop class labels during training
        if torch.rand(1).item() < config.training.p_uncond:
            # Unconditional training
            labels_input = None
        else:
            labels_input = labels

        # Predict noise
        if config.training.use_amp and device.type == 'cuda':
            # Use AMP only on CUDA
            with torch.amp.autocast(device_type='cuda'):
                noise_pred = model(noisy_images, timesteps, labels_input)
                loss = F.mse_loss(noise_pred, noise)
        else:
            # CPU or MPS - no AMP
            noise_pred = model(noisy_images, timesteps, labels_input)
            loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

        optimizer.step()

        # Update EMA
        if config.training.use_ema:
            ema.update()

        # Logging
        total_loss += loss.item()
        current_lr = get_lr(optimizer)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        # Log to WandB
        if use_wandb and batch_idx % config.training.log_every_n_steps == 0:
            step = epoch * num_batches + batch_idx
            wandb.log({
                'train/loss_batch': loss.item(),
                'train/epoch': epoch,
                'learning_rate': current_lr,
            }, step=step)

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    val_loader: DataLoader,
    device: torch.device,
    config: Config,
):
    """
    Validate model.

    Args:
        model: U-Net model
        scheduler: Diffusion scheduler
        val_loader: Validation dataloader
        device: Device
        config: Configuration

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0

    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, scheduler.num_timesteps, (batch_size,), device=device
        ).long()

        noise = torch.randn_like(images)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Always use conditional for validation
        noise_pred = model(noisy_images, timesteps, labels)
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    device: torch.device,
    config: Config,
    num_samples: int = 16,
    use_ema: bool = True,
):
    """
    Generate sample images for visualization.

    Args:
        model: U-Net model
        scheduler: Diffusion scheduler
        device: Device
        config: Configuration
        num_samples: Number of samples to generate
        use_ema: Whether to use EMA weights

    Returns:
        Generated images tensor
    """
    sampler = DDPMSampler(scheduler, model, device)

    # Generate one sample per class
    num_classes = config.model.num_classes
    samples_per_class = max(1, num_samples // num_classes)

    all_samples = []
    for class_idx in range(num_classes):
        class_labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

        samples = sampler.sample(
            batch_size=samples_per_class,
            channels=config.model.in_channels,
            height=config.data.image_size,
            width=config.data.image_size,
            class_labels=class_labels,
            guidance_scale=config.sampling.guidance_scale,
            progress_bar=False,
        )

        all_samples.append(samples)

    # Concatenate and denormalize to [0, 1] for visualization
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    all_samples = denormalize_from_range(all_samples, config.data.normalize_to)
    all_samples = torch.clamp(all_samples, 0, 1)

    return all_samples


def train(config: Config):
    """
    Main training function.

    Args:
        config: Configuration object
    """
    # Print configuration
    config.print_summary()

    # Set seed
    set_seed(config.seed)

    # Create output directories
    paths = create_output_dirs(config.output_dir, config.exp_name)

    # Save config
    config.save(os.path.join(paths['exp_dir'], 'config.json'))

    # Get device
    device = get_device(config.training.device)

    # Initialize WandB
    use_wandb = config.wandb.use_wandb
    if use_wandb:
        # Set WandB environment variables
        os.environ['WANDB_BASE_URL'] = config.wandb.base_url

        # Set API key if available in environment
        if 'WANDB_API_KEY' in os.environ:
            # Already set from .env via load_dotenv()
            pass

        # Check if WANDB_MODE is set (offline/online)
        wandb_mode = os.getenv('WANDB_MODE', 'online')

        print(f"\nWandB Configuration:")
        print(f"  Base URL: {config.wandb.base_url}")
        print(f"  Entity: {config.wandb.entity}")
        print(f"  Project: {config.wandb.project}")
        print(f"  Mode: {wandb_mode}")

        run_name = config.wandb.run_name or f"{config.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            config=config.to_dict(),
        )
        print(f"\nWandB initialized: {config.wandb.project}/{run_name}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Create model
    print("\nCreating model...")
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        channel_mults=config.model.channel_mults,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes,
        embed_dim=config.model.embed_dim,
    ).to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    if use_wandb:
        wandb.log({'model/num_parameters': num_params})
        if config.wandb.watch_model:
            wandb.watch(model, log="all", log_freq=config.wandb.watch_freq, log_graph=True)

    # Create optimizer
    if config.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.eps,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.eps,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    # Create learning rate scheduler
    if config.training.use_scheduler:
        if config.training.scheduler_type == 'cosine':
            scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.epochs
            )
        else:
            scheduler_lr = None
    else:
        scheduler_lr = None

    # Create diffusion scheduler
    diffusion_scheduler = DiffusionScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )

    # Create EMA
    ema = None
    if config.training.use_ema:
        ema = EMA(model, decay=config.training.ema_decay)
        print(f"EMA initialized with decay={config.training.ema_decay}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    best_val_loss = float('inf')

    for epoch in range(1, config.training.epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, diffusion_scheduler, train_loader, optimizer,
            ema, device, epoch, config, use_wandb
        )

        # Validate
        val_loss = validate(model, diffusion_scheduler, val_loader, device, config)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Log to WandB
        if use_wandb:
            wandb.log({
                'train/loss_epoch': train_loss,
                'val/loss': val_loss,
                'epoch': epoch,
            })

        # Update learning rate
        if scheduler_lr is not None:
            scheduler_lr.step()

        # Generate samples
        if epoch % config.training.sample_every_n_epochs == 0:
            print("Generating samples...")

            # Use EMA weights for sampling
            if ema is not None:
                ema.apply_shadow()

            samples = generate_samples(
                model, diffusion_scheduler, device, config,
                num_samples=config.training.num_sample_images,
                use_ema=(ema is not None)
            )

            if ema is not None:
                ema.restore()

            # Log to WandB
            if use_wandb:
                # Create grid of images
                import torchvision
                grid = torchvision.utils.make_grid(samples, nrow=4, padding=2, normalize=False)
                wandb.log({
                    'samples/generated': wandb.Image(grid, caption=f"Epoch {epoch}"),
                    'epoch': epoch,
                })

            # Save samples
            samples_path = os.path.join(paths['samples'], f'epoch_{epoch:04d}.png')
            torchvision.utils.save_image(samples, samples_path, nrow=4)

        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0 or epoch == config.training.epochs:
            checkpoint_path = os.path.join(paths['checkpoints'], f'checkpoint_epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, ema, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(paths['checkpoints'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, ema, best_path)
            print(f"✨ New best model! Val loss: {val_loss:.4f}")

    # Save final model
    final_path = os.path.join(paths['checkpoints'], 'final_model.pth')
    save_checkpoint(model, optimizer, config.training.epochs, val_loss, ema, final_path)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--config', type=str, default='baseline', choices=['small', 'baseline', 'large'],
                        help='Configuration preset')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')

    args = parser.parse_args()

    # Load configuration
    if args.config == 'small':
        config = get_small_config()
    elif args.config == 'baseline':
        config = get_baseline_config()
    elif args.config == 'large':
        from config import get_large_config
        config = get_large_config()
    else:
        config = get_baseline_config()

    # Override with command-line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.no_wandb:
        config.wandb.use_wandb = False
    if args.device != 'auto':
        config.training.device = args.device

    # Run training
    train(config)
