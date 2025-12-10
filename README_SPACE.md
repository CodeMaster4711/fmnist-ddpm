---
title: Fashion-MNIST DDPM Generator
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Fashion-MNIST DDPM Generator

Generate realistic fashion items using a trained Denoising Diffusion Probabilistic Model (DDPM).

## Features

- **10 Fashion Categories**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Classifier-Free Guidance**: Control the strength of class conditioning
- **Batch Generation**: Generate multiple samples at once
- **All Categories View**: Generate samples for all 10 categories simultaneously

## Model Details

This app uses a DDPM trained on the Fashion-MNIST dataset with:
- U-Net architecture with attention
- Cosine noise schedule
- Classifier-free guidance for conditional generation
- EMA (Exponential Moving Average) weights for better quality

## Usage

1. **Select a Category**: Click on any fashion category button to generate samples
2. **Adjust Settings**:
   - Number of Samples: How many images to generate (1-16)
   - Guidance Scale: Higher values = stronger class conditioning (1.0-10.0)
3. **Generate All**: Use the "Generate All Categories" tab to see samples from all categories

## Technical Details

- Model: Small U-Net (32 base channels)
- Training: 150 epochs on Fashion-MNIST
- Diffusion Steps: 500
- Image Size: 28x28 (upscaled to 224x224 for display)

## Credits

Built with:
- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
