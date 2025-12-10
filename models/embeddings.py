"""
Time and Class Embeddings for Diffusion Models

This module implements:
1. Sinusoidal Time Embeddings (like in Transformers)
2. Class Embeddings for conditional generation
"""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for diffusion timesteps.

    Converts integer timesteps into continuous sinusoidal embeddings,
    then projects them to the desired dimension.

    Args:
        embed_dim: Dimension of the time embedding
        max_period: Maximum period for sinusoidal encoding (default: 10000)
    """
    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        # Two-layer MLP to project sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),  # Swish activation (x * sigmoid(x))
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of integer timesteps

        Returns:
            (batch_size, embed_dim) tensor of time embeddings
        """
        # Create sinusoidal embeddings
        half_dim = self.embed_dim // 2

        # Compute frequencies: [max_period^0, max_period^(1/(half_dim-1)), ..., max_period]
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)

        # timesteps: (batch_size, 1), emb: (half_dim,) -> (batch_size, half_dim)
        emb = timesteps.float()[:, None] * emb[None, :]

        # Concatenate sin and cos: (batch_size, embed_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # If embed_dim is odd, pad with zeros
        if self.embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        # Project through MLP
        emb = self.mlp(emb)

        return emb


class ClassEmbedding(nn.Module):
    """
    Class embedding for conditional generation.

    Simple embedding layer that maps class indices to embedding vectors.
    Supports unconditional generation by using a special "null" class.

    Args:
        num_classes: Number of classes
        embed_dim: Dimension of class embeddings
        dropout: Dropout probability for classifier-free guidance
    """
    def __init__(self, num_classes: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout = dropout

        # +1 for unconditional class (null class for classifier-free guidance)
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

        # MLP to project embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, class_labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """
        Args:
            class_labels: (batch_size,) tensor of class indices [0, num_classes-1]
            force_drop: If True, return unconditional embedding (for classifier-free guidance)

        Returns:
            (batch_size, embed_dim) tensor of class embeddings
        """
        batch_size = class_labels.shape[0]

        # During training, randomly drop to unconditional (for classifier-free guidance)
        if self.training and self.dropout > 0:
            # Random mask: True = drop to unconditional
            drop_mask = torch.rand(batch_size, device=class_labels.device) < self.dropout
            # Unconditional class index is num_classes
            class_labels = torch.where(drop_mask, self.num_classes, class_labels)

        # If force_drop, use unconditional class
        if force_drop:
            class_labels = torch.full_like(class_labels, self.num_classes)

        # Embed and project
        emb = self.embedding(class_labels)
        emb = self.mlp(emb)

        return emb


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization (AdaGN).

    Applies Group Normalization, then modulates with scale and shift
    derived from conditioning (time + class embeddings).

    This is the key mechanism for injecting conditioning into the U-Net.

    Args:
        num_channels: Number of channels to normalize
        num_groups: Number of groups for GroupNorm (default: 32)
        embed_dim: Dimension of conditioning embeddings
    """
    def __init__(self, num_channels: int, num_groups: int = 32, embed_dim: int = 512):
        super().__init__()
        self.num_channels = num_channels

        # Group Normalization
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        # Linear projection to predict scale and shift
        self.proj = nn.Linear(embed_dim, num_channels * 2)

        # Initialize projection to output zeros (start with identity transform)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width) feature maps
            emb: (batch_size, embed_dim) conditioning embeddings (time + class)

        Returns:
            (batch_size, num_channels, height, width) normalized and modulated features
        """
        # Normalize
        x = self.norm(x)

        # Get scale and shift from conditioning
        emb = self.proj(emb)  # (batch_size, num_channels * 2)
        scale, shift = emb.chunk(2, dim=1)  # Each: (batch_size, num_channels)

        # Reshape for broadcasting: (batch_size, num_channels, 1, 1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        # Modulate: scale * x + shift (like FiLM)
        # Add 1 to scale to start near identity
        x = (1 + scale) * x + shift

        return x


if __name__ == "__main__":
    # Test the embeddings
    batch_size = 4
    embed_dim = 256
    num_classes = 10

    # Test TimeEmbedding
    time_emb = TimeEmbedding(embed_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    t_emb = time_emb(timesteps)
    print(f"Time embedding shape: {t_emb.shape}")  # Should be (4, 256)
    assert t_emb.shape == (batch_size, embed_dim)

    # Test ClassEmbedding
    class_emb = ClassEmbedding(num_classes, embed_dim, dropout=0.1)
    class_labels = torch.randint(0, num_classes, (batch_size,))
    c_emb = class_emb(class_labels)
    print(f"Class embedding shape: {c_emb.shape}")  # Should be (4, 256)
    assert c_emb.shape == (batch_size, embed_dim)

    # Test unconditional embedding
    c_emb_uncond = class_emb(class_labels, force_drop=True)
    print(f"Unconditional embedding shape: {c_emb_uncond.shape}")

    # Test AdaptiveGroupNorm
    adagn = AdaptiveGroupNorm(num_channels=64, num_groups=32, embed_dim=embed_dim)
    x = torch.randn(batch_size, 64, 28, 28)
    cond = t_emb + c_emb  # Combined conditioning
    x_out = adagn(x, cond)
    print(f"AdaGN output shape: {x_out.shape}")  # Should be (4, 64, 28, 28)
    assert x_out.shape == x.shape

    print("\n✅ All embedding tests passed!")
