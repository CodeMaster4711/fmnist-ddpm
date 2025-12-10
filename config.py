"""
Configuration for Diffusion Model Training

Contains all hyperparameters for:
- Model architecture
- Diffusion process
- Training
- Sampling
- Data loading
"""

from dataclasses import dataclass, field
from typing import List
import json
import os


@dataclass
class ModelConfig:
    """U-Net architecture configuration."""
    in_channels: int = 1  # 1 for Fashion-MNIST (grayscale), 3 for RGB
    out_channels: int = 1
    base_channels: int = 64  # Base number of channels
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # Channel multipliers
    num_res_blocks: int = 2  # ResNet blocks per resolution
    attention_resolutions: List[int] = field(default_factory=lambda: [7])  # Resolutions with attention
    dropout: float = 0.1
    num_classes: int = 10  # Fashion-MNIST classes
    embed_dim: int = 256  # Time and class embedding dimension


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    num_timesteps: int = 1000  # Number of diffusion steps
    beta_schedule: str = 'cosine'  # 'linear' or 'cosine'
    beta_start: float = 1e-4  # For linear schedule
    beta_end: float = 0.02  # For linear schedule


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 0.0

    # Optimizer
    optimizer: str = 'adamw'  # 'adam' or 'adamw'
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', or 'constant'
    warmup_epochs: int = 5

    # Regularization
    gradient_clip: float = 1.0  # Max gradient norm
    ema_decay: float = 0.9999  # Exponential moving average decay
    use_ema: bool = True

    # Classifier-free guidance training
    p_uncond: float = 0.1  # Probability of unconditional training

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision

    # Logging and checkpointing
    log_every_n_steps: int = 10
    sample_every_n_epochs: int = 5  # Generate samples for visualization
    save_every_n_epochs: int = 10  # Save checkpoint
    num_sample_images: int = 16  # Number of images to generate for logging

    # Early stopping
    use_early_stopping: bool = False
    patience: int = 10

    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    num_workers: int = 0  # DataLoader workers (0 to avoid multiprocessing issues)
    pin_memory: bool = False  # Set to False for MPS, True for CUDA


@dataclass
class SamplingConfig:
    """Sampling/generation configuration."""
    guidance_scale: float = 3.0  # Classifier-free guidance strength (1.0 = no guidance)
    num_samples: int = 16  # Number of samples to generate
    batch_size: int = 16  # Batch size for sampling
    save_progressive: bool = False  # Save intermediate denoising steps
    progressive_timesteps: List[int] = field(
        default_factory=lambda: [999, 750, 500, 250, 100, 0]
    )


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = 'fashionmnist'  # 'fashionmnist', 'mnist', 'cifar10'
    data_dir: str = './data'
    image_size: int = 28  # Will resize if needed
    normalize_to: tuple = (-1.0, 1.0)  # Normalize images to this range

    # Augmentation (usually not used for diffusion)
    use_augmentation: bool = False
    horizontal_flip: bool = False


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    use_wandb: bool = True
    project: str = None  # Will be read from env
    entity: str = None  # Will be read from env
    run_name: str = None  # Auto-generated if None
    base_url: str = None  # Will be read from env
    log_model: bool = False  # Log model checkpoints to WandB
    watch_model: bool = True  # Watch gradients and parameters
    watch_freq: int = 100  # Log frequency for model watching

    def __post_init__(self):
        """Load values from environment variables if not set."""
        import os
        if self.base_url is None:
            self.base_url = os.getenv('WANDB_BASE_URL', 'http://localhost:8080')
        if self.entity is None:
            self.entity = os.getenv('WANDB_ENTITY', 'codemaster4711')
        if self.project is None:
            self.project = os.getenv('WANDB_PROJECT', 'diffusion-fashionmnist')


@dataclass
class Config:
    """Complete configuration for diffusion model training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Experiment tracking
    exp_name: str = 'v1_baseline'
    seed: int = 42
    output_dir: str = './outputs'

    def save(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'diffusion': self.diffusion.__dict__,
            'training': self.training.__dict__,
            'sampling': self.sampling.__dict__,
            'data': self.data.__dict__,
            'wandb': self.wandb.__dict__,
            'exp_name': self.exp_name,
            'seed': self.seed,
            'output_dir': self.output_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Load configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict['model']),
            diffusion=DiffusionConfig(**config_dict['diffusion']),
            training=TrainingConfig(**config_dict['training']),
            sampling=SamplingConfig(**config_dict['sampling']),
            data=DataConfig(**config_dict['data']),
            wandb=WandBConfig(**config_dict['wandb']),
            exp_name=config_dict['exp_name'],
            seed=config_dict['seed'],
            output_dir=config_dict['output_dir'],
        )

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        print(f"Config loaded from {path}")
        return cls.from_dict(config_dict)

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"\n📊 Experiment: {self.exp_name}")
        print(f"🌱 Seed: {self.seed}")
        print(f"\n🏗️  Model:")
        print(f"  - Base channels: {self.model.base_channels}")
        print(f"  - Channel mults: {self.model.channel_mults}")
        print(f"  - ResNet blocks: {self.model.num_res_blocks}")
        print(f"  - Attention at: {self.model.attention_resolutions}")
        print(f"  - Dropout: {self.model.dropout}")
        print(f"\n🌀 Diffusion:")
        print(f"  - Timesteps: {self.diffusion.num_timesteps}")
        print(f"  - Beta schedule: {self.diffusion.beta_schedule}")
        print(f"\n🎯 Training:")
        print(f"  - Epochs: {self.training.epochs}")
        print(f"  - Batch size: {self.training.batch_size}")
        print(f"  - Learning rate: {self.training.learning_rate}")
        print(f"  - EMA decay: {self.training.ema_decay}")
        print(f"  - Gradient clip: {self.training.gradient_clip}")
        print(f"  - Use AMP: {self.training.use_amp}")
        print(f"  - Unconditional prob: {self.training.p_uncond}")
        print(f"\n🎨 Sampling:")
        print(f"  - Guidance scale: {self.sampling.guidance_scale}")
        print(f"\n💾 Data:")
        print(f"  - Dataset: {self.data.dataset}")
        print(f"  - Image size: {self.data.image_size}")
        print(f"\n📈 WandB:")
        print(f"  - Enabled: {self.wandb.use_wandb}")
        print(f"  - Project: {self.wandb.project}")
        print(f"  - Entity: {self.wandb.entity}")
        print("=" * 60)


# Predefined configurations

def get_small_config() -> Config:
    """Small model for quick testing."""
    config = Config()
    config.exp_name = 'v1_small'
    config.model.base_channels = 32
    config.model.channel_mults = [1, 2, 4]
    config.model.num_res_blocks = 1
    config.model.attention_resolutions = []
    config.training.epochs = 20
    config.training.batch_size = 256
    config.diffusion.num_timesteps = 500
    return config


def get_baseline_config() -> Config:
    """Baseline configuration for Fashion-MNIST."""
    config = Config()
    config.exp_name = 'v1_baseline'
    config.model.base_channels = 64
    config.model.channel_mults = [1, 2, 4, 8]
    config.model.num_res_blocks = 2
    config.model.attention_resolutions = [7]  # 28->14->7
    config.training.epochs = 50
    config.training.batch_size = 128
    config.diffusion.num_timesteps = 1000
    return config


def get_large_config() -> Config:
    """Large model for best quality."""
    config = Config()
    config.exp_name = 'v1_large'
    config.model.base_channels = 128
    config.model.channel_mults = [1, 2, 3, 4]
    config.model.num_res_blocks = 3
    config.model.attention_resolutions = [7, 14]
    config.training.epochs = 100
    config.training.batch_size = 64
    config.diffusion.num_timesteps = 1000
    return config


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration classes...\n")

    # Create and print baseline config
    config = get_baseline_config()
    config.print_summary()

    # Test save/load
    test_path = '/tmp/test_config.json'
    config.save(test_path)
    loaded_config = Config.load(test_path)

    print("\n\n📋 Testing save/load...")
    assert config.model.base_channels == loaded_config.model.base_channels
    assert config.training.epochs == loaded_config.training.epochs
    print("✅ Save/load test passed!")

    # Print small config
    print("\n\n")
    small_config = get_small_config()
    small_config.print_summary()

    print("\n✅ All config tests passed!")
