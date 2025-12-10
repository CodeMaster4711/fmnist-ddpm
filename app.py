"""
Gradio App for Fashion-MNIST DDPM Image Generation

Interactive interface to generate fashion items using trained diffusion model.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import argparse

from config import Config
from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler
from diffusion.ddpm import DDPMSampler
from utils import EMA, get_device, load_checkpoint, denormalize_from_range


# Fashion-MNIST class names
CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


class FashionDDPMGenerator:
    """Fashion-MNIST DDPM Generator for Gradio."""

    def __init__(self, checkpoint_path: str, config_path: str, use_ema: bool = True):
        """
        Initialize the generator.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config.json
            use_ema: Whether to use EMA weights
        """
        print(f"Loading model from {checkpoint_path}...")

        # Load config
        self.config = Config.load(config_path)

        # Get device
        self.device = get_device('auto')
        print(f"Using device: {self.device}")

        # Create model
        self.model = UNet(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            base_channels=self.config.model.base_channels,
            channel_mults=self.config.model.channel_mults,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=self.config.model.attention_resolutions,
            dropout=0.0,
            num_classes=self.config.model.num_classes,
            embed_dim=self.config.model.embed_dim,
        ).to(self.device)

        # Create EMA
        self.ema = None
        if use_ema:
            self.ema = EMA(self.model, decay=self.config.training.ema_decay)

        # Load checkpoint
        load_checkpoint(self.model, ema=self.ema, path=checkpoint_path, device=self.device)

        # Apply EMA weights
        if self.ema is not None:
            print("Using EMA weights")
            self.ema.apply_shadow()

        # Create scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=self.config.diffusion.num_timesteps,
            beta_schedule=self.config.diffusion.beta_schedule,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
        )

        # Create sampler
        self.sampler = DDPMSampler(self.scheduler, self.model, self.device)

        self.model.eval()
        print("Model loaded successfully!")

    def generate_class(self, class_idx: int, num_samples: int = 4, guidance_scale: float = 3.0):
        """
        Generate samples for a specific class.

        Args:
            class_idx: Class index (0-9)
            num_samples: Number of samples to generate
            guidance_scale: Classifier-free guidance scale

        Returns:
            PIL Image with generated samples in a grid
        """
        with torch.no_grad():
            # Create class labels
            class_labels = torch.full((num_samples,), class_idx, device=self.device, dtype=torch.long)

            # Generate samples
            samples = self.sampler.sample(
                batch_size=num_samples,
                channels=self.config.model.in_channels,
                height=self.config.data.image_size,
                width=self.config.data.image_size,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                progress_bar=True,
            )

            # Denormalize
            samples = denormalize_from_range(samples, self.config.data.normalize_to)
            samples = torch.clamp(samples, 0, 1)

            # Convert to numpy and create grid
            samples_np = samples.cpu().numpy()

            # Create grid (2x2 for 4 samples)
            if num_samples == 1:
                grid = samples_np[0, 0]
            else:
                rows = int(np.sqrt(num_samples))
                cols = (num_samples + rows - 1) // rows

                grid_list = []
                for i in range(rows):
                    row_images = []
                    for j in range(cols):
                        idx = i * cols + j
                        if idx < num_samples:
                            row_images.append(samples_np[idx, 0])
                        else:
                            row_images.append(np.zeros_like(samples_np[0, 0]))
                    grid_list.append(np.concatenate(row_images, axis=1))
                grid = np.concatenate(grid_list, axis=0)

            # Convert to PIL Image
            grid_img = Image.fromarray((grid * 255).astype(np.uint8))

            # Scale up the image for better visibility (28x28 is too small)
            scale_factor = 8  # Scale to 224x224 per image
            new_width = grid_img.width * scale_factor
            new_height = grid_img.height * scale_factor
            grid_img = grid_img.resize((new_width, new_height), Image.NEAREST)

            return grid_img

    def generate_all_classes(self, samples_per_class: int = 4, guidance_scale: float = 3.0):
        """
        Generate samples for all classes.

        Args:
            samples_per_class: Number of samples per class
            guidance_scale: Classifier-free guidance scale

        Returns:
            PIL Image with all classes in a grid
        """
        all_samples = []

        for class_idx in range(10):
            print(f"Generating class {class_idx}: {CLASS_NAMES[class_idx]}...")

            with torch.no_grad():
                class_labels = torch.full((samples_per_class,), class_idx, device=self.device, dtype=torch.long)

                samples = self.sampler.sample(
                    batch_size=samples_per_class,
                    channels=self.config.model.in_channels,
                    height=self.config.data.image_size,
                    width=self.config.data.image_size,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    progress_bar=False,
                )

                # Denormalize
                samples = denormalize_from_range(samples, self.config.data.normalize_to)
                samples = torch.clamp(samples, 0, 1)

                all_samples.append(samples)

        # Stack all samples (10 classes x samples_per_class)
        all_samples = torch.cat(all_samples, dim=0)
        samples_np = all_samples.cpu().numpy()

        # Create grid: 10 rows (classes) x samples_per_class columns
        grid_list = []
        for i in range(10):
            row_images = []
            for j in range(samples_per_class):
                idx = i * samples_per_class + j
                row_images.append(samples_np[idx, 0])
            grid_list.append(np.concatenate(row_images, axis=1))
        grid = np.concatenate(grid_list, axis=0)

        # Convert to PIL Image
        grid_img = Image.fromarray((grid * 255).astype(np.uint8))

        # Scale up the image for better visibility (28x28 is too small)
        scale_factor = 8  # Scale to 224x224 per image
        new_width = grid_img.width * scale_factor
        new_height = grid_img.height * scale_factor
        grid_img = grid_img.resize((new_width, new_height), Image.NEAREST)

        return grid_img


def find_best_checkpoint(model_name: str = None):
    """
    Find the best model checkpoint.

    Args:
        model_name: Specific model to use (e.g., 'v1_small', 'v1_baseline').
                   If None, will search in order of preference.
    """
    if model_name:
        # Check models_release directory first
        release_path = f"models_release/{model_name}_best_model.pth"
        release_config = f"models_release/{model_name}_config.json"

        if os.path.exists(release_path) and os.path.exists(release_config):
            return release_path, release_config

        # Fallback to outputs directory
        path = f"outputs/{model_name}/checkpoints/best_model.pth"
        config_path = f"outputs/{model_name}/config.json"

        if os.path.exists(path) and os.path.exists(config_path):
            return path, config_path
        else:
            raise FileNotFoundError(f"Model {model_name} not found at {path}")

    # Look for best_model.pth in models_release directory first, then outputs
    # Prefer v1_small for faster inference
    possible_paths = [
        ("models_release/v1_small_best_model.pth", "models_release/v1_small_config.json"),
        ("models_release/v1_baseline_best_model.pth", "models_release/v1_baseline_config.json"),
        ("outputs/v1_small/checkpoints/best_model.pth", "outputs/v1_small/config.json"),
        ("outputs/v1_baseline/checkpoints/best_model.pth", "outputs/v1_baseline/config.json"),
        ("outputs/v1_large/checkpoints/best_model.pth", "outputs/v1_large/config.json"),
    ]

    for path, config_path in possible_paths:
        if os.path.exists(path) and os.path.exists(config_path):
            return path, config_path

    raise FileNotFoundError("No checkpoint found! Please train a model first.")


def create_demo(checkpoint_path: str, config_path: str):
    """Create the Gradio demo interface."""

    # Initialize generator
    print("Initializing Fashion-MNIST DDPM Generator...")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    generator = FashionDDPMGenerator(checkpoint_path, config_path, use_ema=True)

    def generate_category(class_idx: int, num_samples: int, guidance_scale: float):
        """Wrapper for Gradio interface."""
        return generator.generate_class(class_idx, num_samples, guidance_scale)

    def generate_all(samples_per_class: int, guidance_scale: float):
        """Wrapper for Gradio interface - generate all classes."""
        return generator.generate_all_classes(samples_per_class, guidance_scale)

    # Create Gradio interface
    with gr.Blocks(title="Fashion-MNIST DDPM Generator") as demo:
        gr.Markdown(
            """
            # Fashion-MNIST DDPM Generator

            Generate realistic fashion items using a trained Denoising Diffusion Probabilistic Model.

            **Select a category** to generate specific items, or **Generate All** to see samples from all categories.
            """
        )

        with gr.Tab("Generate by Category"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    num_samples = gr.Slider(1, 16, value=4, step=1, label="Number of Samples")
                    guidance_scale = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance Scale")
                    gr.Markdown(
                        """
                        **Guidance Scale**: Higher values make generated images more closely match the class.
                        - 1.0: No guidance (unconditional)
                        - 3.0: Recommended (default)
                        - 7.0+: Very strong guidance (may reduce diversity)
                        """
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Select Category")

                    with gr.Row():
                        btn_tshirt = gr.Button("👕 T-shirt/top", variant="primary")
                        btn_trouser = gr.Button("👖 Trouser", variant="primary")
                        btn_pullover = gr.Button("🧥 Pullover", variant="primary")

                    with gr.Row():
                        btn_dress = gr.Button("👗 Dress", variant="primary")
                        btn_coat = gr.Button("🧥 Coat", variant="primary")
                        btn_sandal = gr.Button("👡 Sandal", variant="primary")

                    with gr.Row():
                        btn_shirt = gr.Button("👔 Shirt", variant="primary")
                        btn_sneaker = gr.Button("👟 Sneaker", variant="primary")
                        btn_bag = gr.Button("👜 Bag", variant="primary")

                    with gr.Row():
                        btn_boot = gr.Button("👢 Ankle boot", variant="primary")

            output_category = gr.Image(label="Generated Images", type="pil")

            # Connect buttons
            btn_tshirt.click(fn=lambda n, g: generate_category(0, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_trouser.click(fn=lambda n, g: generate_category(1, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_pullover.click(fn=lambda n, g: generate_category(2, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_dress.click(fn=lambda n, g: generate_category(3, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_coat.click(fn=lambda n, g: generate_category(4, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_sandal.click(fn=lambda n, g: generate_category(5, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_shirt.click(fn=lambda n, g: generate_category(6, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_sneaker.click(fn=lambda n, g: generate_category(7, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_bag.click(fn=lambda n, g: generate_category(8, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)
            btn_boot.click(fn=lambda n, g: generate_category(9, n, g), inputs=[num_samples, guidance_scale], outputs=output_category)

        with gr.Tab("Generate All Categories"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    samples_per_class_all = gr.Slider(1, 8, value=4, step=1, label="Samples per Class")
                    guidance_scale_all = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance Scale")

                    btn_generate_all = gr.Button("🎨 Generate All Categories", variant="primary", size="lg")

                    gr.Markdown(
                        """
                        This will generate samples for all 10 Fashion-MNIST categories.
                        The result will be a grid with one row per category.

                        ⚠️ This may take a while depending on your hardware!
                        """
                    )

                with gr.Column(scale=2):
                    output_all = gr.Image(label="All Categories", type="pil")

            btn_generate_all.click(
                fn=generate_all,
                inputs=[samples_per_class_all, guidance_scale_all],
                outputs=output_all
            )

        gr.Markdown(
            """
            ---
            ### About

            This app uses a Denoising Diffusion Probabilistic Model (DDPM) trained on the Fashion-MNIST dataset.
            The model learns to generate realistic fashion items through a diffusion process.

            **Classes:**
            - 0: T-shirt/top
            - 1: Trouser
            - 2: Pullover
            - 3: Dress
            - 4: Coat
            - 5: Sandal
            - 6: Shirt
            - 7: Sneaker
            - 8: Bag
            - 9: Ankle boot
            """
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion-MNIST DDPM Gradio App')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (e.g., outputs/v1_small/checkpoints/best_model.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (e.g., outputs/v1_small/config.json)')
    parser.add_argument('--model', type=str, default='v1_small',
                        help='Model name to use (e.g., v1_small, v1_baseline)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the app on')

    args = parser.parse_args()

    # Determine checkpoint and config paths
    if args.checkpoint and args.config:
        checkpoint_path = args.checkpoint
        config_path = args.config
    else:
        checkpoint_path, config_path = find_best_checkpoint(args.model)

    # Create and launch demo
    demo = create_demo(checkpoint_path, config_path)
    demo.launch(share=args.share, server_port=args.port)
