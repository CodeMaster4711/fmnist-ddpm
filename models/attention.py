"""
Self-Attention Blocks for Diffusion U-Net

Implements multi-head self-attention for spatial feature learning.
This helps the model capture long-range dependencies in images.
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention block for spatial features.

    Applies self-attention across spatial positions (H×W) in feature maps.
    Uses multi-head attention like in Transformers.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.0)
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(head_dim)

        # Layer norm before attention (pre-norm architecture)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)

        # Query, Key, Value projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)

        Returns:
            (batch_size, channels, height, width) with self-attention applied
        """
        batch_size, channels, height, width = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, H*W, head_dim)

        # Compute attention scores
        # (B, num_heads, H*W, head_dim) @ (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, H*W)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Softmax over keys dimension
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # (B, num_heads, H*W, H*W) @ (B, num_heads, H*W, head_dim) -> (B, num_heads, H*W, head_dim)
        out = torch.matmul(attn, v)

        # Reshape back to image format
        out = out.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, H*W)
        out = out.reshape(batch_size, channels, height, width)

        # Output projection
        out = self.proj(out)

        # Residual connection
        return out + residual


class AttentionBlock(nn.Module):
    """
    Complete attention block with normalization and feedforward network.

    This is a simplified version of a Transformer block adapted for spatial features.
    Includes:
    - Multi-head self-attention
    - Feed-forward network (optional)
    - Residual connections

    Args:
        channels: Number of channels
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_ffn: Whether to include feedforward network (default: False)
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_ffn: bool = False
    ):
        super().__init__()
        self.channels = channels

        # Self-attention
        self.attn = MultiHeadSelfAttention(channels, num_heads, dropout)

        # Optional feedforward network
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ffn = nn.GroupNorm(num_groups=32, num_channels=channels)
            self.ffn = nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(channels * 4, channels, kernel_size=1),
            )
            # Initialize FFN output to zero for residual path
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)

        Returns:
            (batch_size, channels, height, width) with attention applied
        """
        # Self-attention with residual
        x = self.attn(x)

        # Optional feedforward network
        if self.use_ffn:
            residual = x
            x = self.norm_ffn(x)
            x = self.ffn(x)
            x = x + residual

        return x


class LinearAttention(nn.Module):
    """
    Linear (efficient) self-attention block.

    Uses linear attention approximation for faster computation.
    Complexity: O(N) instead of O(N²) where N = H×W.

    This is useful for higher resolutions where standard attention becomes expensive.

    Args:
        channels: Number of channels
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: 32)
    """
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        hidden_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)

        # Projections
        self.qkv = nn.Conv2d(channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)

        Returns:
            (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        residual = x

        x = self.norm(x)

        # Q, K, V projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, self.num_heads * 3, self.head_dim, height * width)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Softmax on queries and keys separately (linear attention trick)
        q = torch.softmax(q, dim=-1)
        k = torch.softmax(k, dim=-2)

        # Compute attention: (B, heads, dim, HW) @ (B, heads, HW, dim) = (B, heads, dim, dim)
        context = torch.matmul(k, v.transpose(-2, -1))

        # Apply to queries: (B, heads, dim, HW) @ (B, heads, dim, dim) = (B, heads, dim, HW)
        out = torch.matmul(context.transpose(-2, -1), q)

        # Reshape back
        out = out.reshape(batch_size, self.num_heads * self.head_dim, height, width)

        # Project back to channels
        out = self.proj(out)

        return out + residual


if __name__ == "__main__":
    # Test attention blocks
    batch_size = 2
    channels = 64
    height, width = 28, 28

    x = torch.randn(batch_size, channels, height, width)

    # Test MultiHeadSelfAttention
    print("Testing MultiHeadSelfAttention...")
    attn = MultiHeadSelfAttention(channels, num_heads=4)
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Test AttentionBlock
    print("\nTesting AttentionBlock...")
    attn_block = AttentionBlock(channels, num_heads=4, use_ffn=True)
    out = attn_block(x)
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Test LinearAttention
    print("\nTesting LinearAttention...")
    linear_attn = LinearAttention(channels, num_heads=4, head_dim=32)
    out = linear_attn(x)
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Memory test
    print("\nMemory comparison (full vs linear attention):")
    import time

    # Full attention
    x_large = torch.randn(1, 128, 64, 64)  # Larger resolution
    full_attn = MultiHeadSelfAttention(128, num_heads=8)

    start = time.time()
    with torch.no_grad():
        _ = full_attn(x_large)
    full_time = time.time() - start
    print(f"Full attention time: {full_time:.4f}s")

    # Linear attention
    lin_attn = LinearAttention(128, num_heads=8)
    start = time.time()
    with torch.no_grad():
        _ = lin_attn(x_large)
    lin_time = time.time() - start
    print(f"Linear attention time: {lin_time:.4f}s")
    print(f"Speedup: {full_time / lin_time:.2f}x")

    print("\n✅ All attention tests passed!")
