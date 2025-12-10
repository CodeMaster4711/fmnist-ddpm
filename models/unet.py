"""
U-Net Architecture for Diffusion Models

Implements a U-Net with:
- Encoder (downsampling path)
- Decoder (upsampling path)
- Skip connections
- Time and class conditioning via AdaGN
- Self-attention at specified resolutions
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .embeddings import TimeEmbedding, ClassEmbedding, AdaptiveGroupNorm
from .attention import AttentionBlock, LinearAttention


class ResNetBlock(nn.Module):
    """
    ResNet block with Adaptive Group Normalization for conditioning.

    Structure:
    - AdaGN + SiLU + Conv
    - AdaGN + SiLU + Dropout + Conv
    - Skip connection (with projection if channels change)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        embed_dim: Dimension of conditioning embeddings (time + class)
        dropout: Dropout probability
        num_groups: Number of groups for GroupNorm
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First norm + conv
        self.norm1 = AdaptiveGroupNorm(in_channels, num_groups, embed_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second norm + conv
        self.norm2 = AdaptiveGroupNorm(out_channels, num_groups, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, height, width)
            emb: (batch_size, embed_dim) conditioning embeddings

        Returns:
            (batch_size, out_channels, height, width)
        """
        residual = x

        # First block
        x = self.norm1(x, emb)
        x = self.act(x)
        x = self.conv1(x)

        # Second block
        x = self.norm2(x, emb)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # Skip connection
        return x + self.skip(residual)


class DownBlock(nn.Module):
    """
    Downsampling block in U-Net encoder.

    Contains:
    - Multiple ResNet blocks
    - Optional attention
    - Downsampling (strided conv or avg pool)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        embed_dim: Dimension of conditioning embeddings
        num_res_blocks: Number of ResNet blocks
        dropout: Dropout probability
        use_attention: Whether to use self-attention
        downsample: Whether to downsample at the end
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        downsample: bool = True,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.use_attention = use_attention
        self.downsample = downsample

        # ResNet blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResNetBlock(in_ch, out_channels, embed_dim, dropout)
            )

        # Optional attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads=4, dropout=dropout)
        else:
            self.attn = None

        # Downsampling
        if downsample:
            # Use strided convolution for downsampling
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_conv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, in_channels, height, width)
            emb: (batch_size, embed_dim) conditioning

        Returns:
            x: (batch_size, out_channels, height//2, width//2) if downsample else same resolution
            skip: Skip connection to be used in decoder (features before downsampling)
        """
        # ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x, emb)

        # Attention
        if self.attn is not None:
            x = self.attn(x)

        # Save skip connection before downsampling
        skip = x

        # Downsample
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)

        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block in U-Net decoder.

    Contains:
    - Upsampling (transposed conv or interpolation + conv)
    - Multiple ResNet blocks (with skip connections from encoder)
    - Optional attention

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        embed_dim: Dimension of conditioning embeddings
        num_res_blocks: Number of ResNet blocks
        dropout: Dropout probability
        use_attention: Whether to use self-attention
        upsample: Whether to upsample at the beginning
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        upsample: bool = True,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.use_attention = use_attention
        self.upsample = upsample

        # Upsampling
        if upsample:
            # Use transposed convolution for upsampling
            self.upsample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample_conv = None

        # ResNet blocks (first block receives skip connection, so 2x channels)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # First block receives concatenated skip connection
            in_ch = in_channels * 2 if i == 0 else out_channels
            self.res_blocks.append(
                ResNetBlock(in_ch, out_channels, embed_dim, dropout)
            )

        # Optional attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads=4, dropout=dropout)
        else:
            self.attn = None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, height, width)
            skip: Skip connection from encoder (target resolution)
            emb: (batch_size, embed_dim) conditioning

        Returns:
            (batch_size, out_channels, height*2, width*2) if upsample else same resolution
        """
        # Upsample
        if self.upsample_conv is not None:
            x = self.upsample_conv(x)

        # Match spatial dimensions with skip connection (fix size mismatches from downsampling)
        if x.shape[2:] != skip.shape[2:]:
            # Interpolate to match skip connection size
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode='nearest'
            )

        # Concatenate skip connection before first ResBlock
        x = torch.cat([x, skip], dim=1)

        # ResNet blocks (first one processes concatenated features)
        for res_block in self.res_blocks:
            x = res_block(x, emb)

        # Attention
        if self.attn is not None:
            x = self.attn(x)

        return x


class UNet(nn.Module):
    """
    U-Net for Diffusion Models with time and class conditioning.

    Architecture:
    - Initial convolution
    - Encoder (downsampling) with skip connections
    - Middle block with attention
    - Decoder (upsampling) with skip connections
    - Output convolution

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (usually same as in_channels)
        base_channels: Base number of channels (default: 64)
        channel_mults: Channel multipliers for each resolution (default: [1, 2, 4, 8])
        num_res_blocks: Number of ResNet blocks per resolution (default: 2)
        attention_resolutions: Resolutions where to apply attention (default: [8, 16])
        dropout: Dropout probability (default: 0.1)
        num_classes: Number of classes for class-conditional generation (default: 10)
        embed_dim: Dimension of time and class embeddings (default: 256)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8, 16],
        dropout: float = 0.1,
        num_classes: int = 10,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        # Time and class embeddings
        self.time_embed = TimeEmbedding(embed_dim)
        self.class_embed = ClassEmbedding(num_classes, embed_dim, dropout=0.1)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        channels = [base_channels] + [base_channels * m for m in channel_mults]

        for i in range(len(channel_mults)):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Assume input resolution is 28x28, calculate current resolution
            resolution = 28 // (2 ** i)
            use_attn = resolution in attention_resolutions

            # Don't downsample on last block
            downsample = (i < len(channel_mults) - 1)

            self.down_blocks.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    embed_dim=embed_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=use_attn,
                    downsample=downsample,
                )
            )

        # Middle block (bottleneck)
        mid_channels = channels[-1]
        self.mid_block = nn.ModuleList([
            ResNetBlock(mid_channels, mid_channels, embed_dim, dropout),
            AttentionBlock(mid_channels, num_heads=4, dropout=dropout),
            ResNetBlock(mid_channels, mid_channels, embed_dim, dropout),
        ])

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        channels_reversed = list(reversed(channels))

        for i in range(len(channel_mults)):
            in_ch = channels_reversed[i]
            out_ch = channels_reversed[i + 1]

            # Calculate resolution
            resolution = 28 // (2 ** (len(channel_mults) - 1 - i))
            use_attn = resolution in attention_resolutions

            # Don't upsample on first block
            upsample = (i > 0)

            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    embed_dim=embed_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    use_attention=use_attn,
                    upsample=upsample,
                )
            )

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # Initialize output layer to zero (start near identity)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        force_drop_class: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: (batch_size, in_channels, height, width) noisy input
            timesteps: (batch_size,) diffusion timesteps
            class_labels: (batch_size,) class labels for conditioning (optional)
            force_drop_class: If True, use unconditional generation

        Returns:
            (batch_size, out_channels, height, width) predicted noise
        """
        batch_size = x.shape[0]

        # Get embeddings
        t_emb = self.time_embed(timesteps)  # (B, embed_dim)

        if class_labels is not None:
            c_emb = self.class_embed(class_labels, force_drop=force_drop_class)
        else:
            # Unconditional (use null class)
            c_emb = self.class_embed(torch.zeros(batch_size, dtype=torch.long, device=x.device), force_drop=True)

        # Combine time and class embeddings
        emb = t_emb + c_emb  # (B, embed_dim)

        # Initial convolution
        x = self.conv_in(x)

        # Encoder
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, emb)
            skip_connections.append(skip)

        # Middle
        for layer in self.mid_block:
            if isinstance(layer, ResNetBlock):
                x = layer(x, emb)
            else:
                x = layer(x)

        # Decoder (reverse order, matching encoder levels)
        for up_block in self.up_blocks:
            # Pop skip connection from end (matches encoder level)
            skip = skip_connections.pop()
            x = up_block(x, skip, emb)

        # Output
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    # Test U-Net
    print("Testing U-Net...")

    batch_size = 2
    in_channels = 1
    height, width = 28, 28
    num_classes = 10

    # Create model
    model = UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=32,  # Small for testing
        channel_mults=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[7],  # 28 -> 14 -> 7
        dropout=0.1,
        num_classes=num_classes,
        embed_dim=128,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    class_labels = torch.randint(0, num_classes, (batch_size,))

    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"Class labels: {class_labels}")

    # Forward pass
    with torch.no_grad():
        output = model(x, timesteps, class_labels)

    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {output.shape}"

    # Test unconditional generation
    print("\nTesting unconditional generation...")
    with torch.no_grad():
        output_uncond = model(x, timesteps, class_labels, force_drop_class=True)
    print(f"Unconditional output shape: {output_uncond.shape}")

    # Test without class labels
    print("\nTesting without class labels...")
    with torch.no_grad():
        output_no_class = model(x, timesteps, class_labels=None)
    print(f"Output shape (no class): {output_no_class.shape}")

    print("\n✅ All U-Net tests passed!")
