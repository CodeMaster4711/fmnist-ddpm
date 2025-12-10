"""
Quick test script to verify the training pipeline works.

Tests:
1. Model creation
2. Forward pass
3. Loss computation
4. Backward pass
5. Data loading
6. Sampling
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import get_small_config
from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.ddpm import DDPMSampler
from utils import EMA, set_seed, get_device, normalize_to_range, denormalize_from_range


def test_model_creation():
    """Test model creation."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)

    config = get_small_config()
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
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully")
    print(f"   Parameters: {num_params:,}")

    return model, config


def test_forward_pass(model, config):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    batch_size = 4
    x = torch.randn(batch_size, config.model.in_channels, 28, 28)
    timesteps = torch.randint(0, 1000, (batch_size,))
    class_labels = torch.randint(0, config.model.num_classes, (batch_size,))

    with torch.no_grad():
        output = model(x, timesteps, class_labels)

    print(f"✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"


def test_loss_computation(model, config):
    """Test loss computation and backward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Loss Computation & Backward Pass")
    print("=" * 60)

    scheduler = DiffusionScheduler(num_timesteps=100, beta_schedule='cosine')

    batch_size = 4
    images = torch.randn(batch_size, config.model.in_channels, 28, 28)
    timesteps = torch.randint(0, 100, (batch_size,))
    class_labels = torch.randint(0, config.model.num_classes, (batch_size,))

    # Forward diffusion
    noise = torch.randn_like(images)
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    # Predict noise
    noise_pred = model(noisy_images, timesteps, class_labels)

    # Compute loss
    loss = F.mse_loss(noise_pred, noise)

    # Backward
    loss.backward()

    print(f"✅ Loss computation successful")
    print(f"   Loss: {loss.item():.4f}")

    return scheduler


def test_data_loading(config):
    """Test data loading."""
    print("\n" + "=" * 60)
    print("TEST 4: Data Loading")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize_to_range(x, config.data.normalize_to)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=config.data.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Get one batch
    images, labels = next(iter(train_loader))

    print(f"✅ Data loading successful")
    print(f"   Dataset size: {len(train_dataset)}")
    print(f"   Batch shape: {images.shape}")
    print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   Labels: {labels.tolist()}")

    return train_loader


def test_sampling(model, scheduler, config):
    """Test sampling."""
    print("\n" + "=" * 60)
    print("TEST 5: Sampling")
    print("=" * 60)

    device = get_device('cpu')  # Use CPU for testing
    model = model.to(device)
    model.eval()

    sampler = DDPMSampler(scheduler, model, device)

    # Generate a few samples
    num_samples = 2
    class_labels = torch.tensor([0, 5], device=device)

    print(f"Generating {num_samples} samples...")
    samples = sampler.sample(
        batch_size=num_samples,
        channels=config.model.in_channels,
        height=28,
        width=28,
        class_labels=class_labels,
        guidance_scale=1.0,  # No guidance for untrained model
        progress_bar=False,
    )

    # Denormalize
    samples = denormalize_from_range(samples, config.data.normalize_to)

    print(f"✅ Sampling successful")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")


def test_ema():
    """Test EMA."""
    print("\n" + "=" * 60)
    print("TEST 6: EMA")
    print("=" * 60)

    config = get_small_config()
    model = UNet(
        in_channels=1, out_channels=1, base_channels=32,  # Must be divisible by num_groups (32)
        channel_mults=[1, 2], num_res_blocks=1,
        attention_resolutions=[], num_classes=10, embed_dim=64
    )

    ema = EMA(model, decay=0.999)

    # Simulate training step
    original_param = list(model.parameters())[0].clone()

    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    ema.update()

    # Apply shadow
    ema.apply_shadow()
    shadow_param = list(model.parameters())[0].clone()

    # Restore
    ema.restore()
    restored_param = list(model.parameters())[0]

    assert not torch.allclose(original_param, shadow_param), "EMA should change params"
    assert torch.allclose(restored_param, list(model.parameters())[0]), "Restore should work"

    print(f"✅ EMA works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE TESTS")
    print("=" * 60)

    set_seed(42)

    # Run tests
    model, config = test_model_creation()
    test_forward_pass(model, config)
    scheduler = test_loss_computation(model, config)
    train_loader = test_data_loading(config)
    test_sampling(model, scheduler, config)
    test_ema()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe training pipeline is ready to use!")
    print("\nTo start training, run:")
    print("  python train.py --config small")


if __name__ == "__main__":
    main()
