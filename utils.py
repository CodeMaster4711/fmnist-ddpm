"""
Utility functions for diffusion model training

Includes:
- Exponential Moving Average (EMA) for model parameters
- Seed setting for reproducibility
- Device selection
- Image preprocessing and postprocessing
"""

import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from typing import Optional
import os


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a moving average of model weights, which often produces
    better samples than the final trained weights.

    Usage:
        ema = EMA(model, decay=0.9999)

        # During training
        loss.backward()
        optimizer.step()
        ema.update(model)  # Update EMA weights

        # For sampling
        ema.apply_shadow()  # Use EMA weights
        samples = model.sample()
        ema.restore()  # Restore original weights

    Args:
        model: PyTorch model to track
        decay: Decay rate for EMA (default: 0.9999)
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        self.register()

    def register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: Optional[nn.Module] = None):
        """
        Update EMA weights.

        Args:
            model: Model to update from (uses self.model if None)
        """
        if model is not None:
            self.model = model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow dict"
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply EMA weights to model.

        Saves original weights in backup and loads EMA weights.
        Call restore() to revert.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = 'auto') -> torch.device:
    """
    Get PyTorch device.

    Args:
        device: 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        torch.device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    device = torch.device(device)
    print(f"Using device: {device}")

    return device


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_to_range(images: torch.Tensor, target_range: tuple = (-1.0, 1.0)) -> torch.Tensor:
    """
    Normalize images to target range.

    Assumes input images are in [0, 1] range (from torchvision.transforms.ToTensor()).

    Args:
        images: (batch_size, channels, height, width) images in [0, 1]
        target_range: Target (min, max) range

    Returns:
        Normalized images in target range
    """
    min_val, max_val = target_range
    # [0, 1] -> [min_val, max_val]
    return images * (max_val - min_val) + min_val


def denormalize_from_range(images: torch.Tensor, source_range: tuple = (-1.0, 1.0)) -> torch.Tensor:
    """
    Denormalize images from source range back to [0, 1].

    Args:
        images: Images in source range
        source_range: Source (min, max) range

    Returns:
        Images in [0, 1] range
    """
    min_val, max_val = source_range
    # [min_val, max_val] -> [0, 1]
    return (images - min_val) / (max_val - min_val)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    ema: Optional[EMA] = None,
    path: str = 'checkpoint.pth'
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        ema: EMA object (optional)
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None,
    path: str = 'checkpoint.pth',
    device: torch.device = None
) -> dict:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        ema: EMA object to load state into (optional)
        path: Checkpoint path
        device: Device to load to

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return checkpoint


def create_output_dirs(base_dir: str, exp_name: str) -> dict:
    """
    Create output directories for experiment.

    Creates:
    - base_dir/exp_name/checkpoints
    - base_dir/exp_name/samples
    - base_dir/exp_name/logs

    Args:
        base_dir: Base output directory
        exp_name: Experiment name

    Returns:
        Dictionary with paths
    """
    exp_dir = os.path.join(base_dir, exp_name)
    paths = {
        'exp_dir': exp_dir,
        'checkpoints': os.path.join(exp_dir, 'checkpoints'),
        'samples': os.path.join(exp_dir, 'samples'),
        'logs': os.path.join(exp_dir, 'logs'),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    print(f"Created output directories at {exp_dir}")
    return paths


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def warmup_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    current_lr: float
) -> float:
    """
    Compute learning rate with linear warmup.

    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        base_lr: Target learning rate after warmup
        current_lr: Current learning rate

    Returns:
        New learning rate
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return current_lr


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...\n")

    # Test seed setting
    print("Testing seed setting...")
    set_seed(42)
    x1 = torch.randn(10)
    set_seed(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2), "Seed setting failed!"
    print("✅ Seed setting works")

    # Test device selection
    print("\nTesting device selection...")
    device = get_device('auto')
    print(f"Auto-selected device: {device}")

    # Test EMA
    print("\nTesting EMA...")
    from models.unet import UNet
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[],
    )

    ema = EMA(model, decay=0.999)

    # Simulate training
    original_param = list(model.parameters())[0].clone()

    # Update model
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Update EMA
    ema.update()

    # Check that EMA weights are different from current weights
    current_param = list(model.parameters())[0]
    assert not torch.allclose(current_param, original_param), "Model should have changed"

    # Apply shadow
    ema.apply_shadow()
    shadow_param = list(model.parameters())[0].clone()

    # Restore
    ema.restore()
    restored_param = list(model.parameters())[0]

    assert torch.allclose(restored_param, current_param), "Restore failed"
    assert not torch.allclose(shadow_param, current_param), "Shadow should be different"

    print("✅ EMA works correctly")

    # Test normalization
    print("\nTesting normalization...")
    images = torch.rand(4, 1, 28, 28)  # [0, 1]
    normalized = normalize_to_range(images, (-1, 1))
    denormalized = denormalize_from_range(normalized, (-1, 1))

    assert normalized.min() >= -1.0 and normalized.max() <= 1.0, "Normalization failed"
    assert torch.allclose(images, denormalized, atol=1e-6), "Denormalization failed"
    print("✅ Normalization works")

    # Test parameter counting
    print("\nTesting parameter counting...")
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} parameters")
    assert num_params > 0, "Parameter counting failed"
    print("✅ Parameter counting works")

    # Test checkpoint save/load
    print("\nTesting checkpoint save/load...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, epoch=10, loss=0.5, ema=ema, path='/tmp/test_checkpoint.pth')

    # Create new model and load
    model2 = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[],
    )
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ema2 = EMA(model2, decay=0.999)

    checkpoint = load_checkpoint(model2, optimizer2, ema2, path='/tmp/test_checkpoint.pth')
    assert checkpoint['epoch'] == 10, "Checkpoint loading failed"
    print("✅ Checkpoint save/load works")

    print("\n✅ All utility tests passed!")
