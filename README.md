# Diffusion Model v1 - Baseline DDPM

Vollständige Implementierung eines **Denoising Diffusion Probabilistic Model (DDPM)** für Fashion-MNIST.

## 🏗️ Architektur

### Implementierte Komponenten

1. **U-Net** ([models/unet.py](models/unet.py))
   - Encoder-Decoder Architektur mit Skip Connections
   - ResNet Blocks mit Adaptive Group Normalization
   - Self-Attention bei definierten Auflösungen
   - Time und Class Conditioning

2. **Time & Class Embeddings** ([models/embeddings.py](models/embeddings.py))
   - Sinusoidal Time Embeddings (wie in Transformers)
   - Learnable Class Embeddings
   - Adaptive Group Normalization (AdaGN) für Conditioning

3. **Self-Attention** ([models/attention.py](models/attention.py))
   - Multi-Head Self-Attention für räumliche Features
   - Linear Attention (effizient für höhere Auflösungen)
   - Attention Block mit Feed-Forward Network

4. **Diffusion Process** ([diffusion/scheduler.py](diffusion/scheduler.py))
   - Forward Process (Noise hinzufügen)
   - Cosine und Linear Beta Schedules
   - Posterior-Berechnungen für Reverse Process

5. **DDPM Sampler** ([diffusion/ddpm.py](diffusion/ddpm.py))
   - Iteratives Denoising von Rauschen zu Bildern
   - Classifier-Free Guidance
   - Progressive Denoising Visualisierung

## 📦 Installation

```bash
# In das Projekt-Root-Verzeichnis wechseln
cd /path/to/DataScienceTutorial

# Dependencies installieren
pip install -r requirements.txt

# WandB Server starten (lokal)
cd Imagenette
docker-compose up -d
cd ..
```

## 🚀 Training

### Quick Start (Small Config - zum Testen)

```bash
cd deffusion/v1
python train.py --config small --epochs 20
```

**Small Config:**
- Base Channels: 32
- Channel Mults: [1, 2, 4]
- ResNet Blocks: 1
- Timesteps: 500
- Epochs: 20
- ~10-15 Minuten auf M-Series Mac

### Baseline Config (Empfohlen)

```bash
python train.py --config baseline
```

**Baseline Config:**
- Base Channels: 64
- Channel Mults: [1, 2, 4, 8]
- ResNet Blocks: 2
- Attention: Auflösung 7
- Timesteps: 1000
- Epochs: 50
- ~30-60 Minuten auf M-Series Mac

### Large Config (Beste Qualität)

```bash
python train.py --config large
```

**Large Config:**
- Base Channels: 128
- Channel Mults: [1, 2, 3, 4]
- ResNet Blocks: 3
- Attention: Auflösungen 7, 14
- Timesteps: 1000
- Epochs: 100
- ~2-3 Stunden auf M-Series Mac

### Optionen

```bash
# Training ohne WandB
python train.py --config baseline --no-wandb

# Custom Hyperparameter
python train.py --config baseline --epochs 100 --batch_size 256 --lr 0.0001

# Spezifisches Device
python train.py --config baseline --device cuda  # oder 'mps' für Mac
```

## 🎨 Sampling (Bildgenerierung)

### Nach dem Training

```bash
# Einfaches Grid (16 Bilder)
python sample.py \
    --checkpoint outputs/v1_baseline/checkpoints/best_model.pth \
    --config outputs/v1_baseline/config.json \
    --num_samples 16 \
    --guidance_scale 3.0 \
    --mode grid

# Class-konditionierte Generierung
python sample.py \
    --checkpoint outputs/v1_baseline/checkpoints/best_model.pth \
    --config outputs/v1_baseline/config.json \
    --class_label 5 \
    --num_samples 16 \
    --guidance_scale 3.0

# Alle Klassen (Grid mit einer Zeile pro Klasse)
python sample.py \
    --checkpoint outputs/v1_baseline/checkpoints/best_model.pth \
    --config outputs/v1_baseline/config.json \
    --mode class_grid \
    --guidance_scale 3.0

# Progressive Denoising Visualisierung
python sample.py \
    --checkpoint outputs/v1_baseline/checkpoints/best_model.pth \
    --config outputs/v1_baseline/config.json \
    --mode progressive \
    --class_label 3 \
    --guidance_scale 3.0

# Mit EMA Weights (bessere Qualität)
python sample.py \
    --checkpoint outputs/v1_baseline/checkpoints/best_model.pth \
    --config outputs/v1_baseline/config.json \
    --use_ema \
    --guidance_scale 3.0
```

## 📊 WandB Monitoring

Das Training loggt automatisch zu WandB (lokal):

**URL:** http://localhost:8080
**Projekt:** diffusion-fashionmnist
**Entity:** codemaster4711

**Geloggte Metriken:**
- Train Loss (per Batch und Epoch)
- Validation Loss
- Learning Rate
- Gradients & Parameter (optional)
- Generated Samples (alle N Epochen)
- Model Architecture Graph

## 🧪 Tests

```bash
# Pipeline-Tests durchführen
python test_pipeline.py
```

Testet:
- Model Creation
- Forward Pass
- Loss Computation
- Data Loading
- Sampling
- EMA

## 📁 Projekt-Struktur

```
deffusion/v1/
├── models/
│   ├── __init__.py
│   ├── embeddings.py      # Time & Class Embeddings, AdaGN
│   ├── attention.py       # Self-Attention Blocks
│   └── unet.py           # U-Net Architektur
│
├── diffusion/
│   ├── __init__.py
│   ├── scheduler.py      # Diffusion Scheduler (Beta Schedule, Forward Process)
│   └── ddpm.py          # DDPM Sampler (Reverse Process)
│
├── config.py            # Konfiguration & Hyperparameter
├── utils.py             # EMA, Seed Setting, Device Management
├── train.py             # Training Script
├── sample.py            # Sampling Script
├── test_pipeline.py     # Tests
└── README.md           # Diese Datei
```

## 🎯 Fashion-MNIST Klassen

```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

## 💡 Tipps

### Training
- **Guidance Scale:** 3.0 ist ein guter Startpunkt. Höhere Werte (5.0-7.0) geben schärfere, aber weniger diverse Bilder.
- **EMA:** Unbedingt verwenden! EMA-Weights produzieren deutlich bessere Samples.
- **Batch Size:** Größer ist besser für Stabilität, aber limitiert durch GPU Memory.
- **Learning Rate:** 2e-4 funktioniert gut. Bei Instabilität auf 1e-4 reduzieren.

### Sampling
- Verwenden Sie `--use_ema` für beste Qualität
- Progressive Denoising ist nützlich zum Verständnis des Prozesses
- Class-conditional mit Guidance Scale > 1.0 für bessere class-spezifische Features

### WandB
- Samples werden während des Trainings alle 5 Epochen generiert
- Vergleichen Sie verschiedene Runs anhand der Val Loss
- Gradient Tracking kann bei langsamen Machines deaktiviert werden: `config.wandb.watch_model = False`

## 📈 Erwartete Ergebnisse

**Nach 50 Epochen (Baseline Config):**
- Train Loss: ~0.01-0.03
- Val Loss: ~0.01-0.03
- Sample Qualität: Erkennbare Fashion-MNIST Items, etwas verschwommen
- Training Zeit: ~30-60 Minuten (M-Series Mac / GPU)

**Nach 100 Epochen (Large Config):**
- Train Loss: <0.01
- Val Loss: <0.01
- Sample Qualität: Scharfe, realistische Fashion-MNIST Items
- Training Zeit: ~2-3 Stunden

## 🔧 Troubleshooting

### "CUDA out of memory"
```bash
# Batch Size reduzieren
python train.py --config baseline --batch_size 64

# Kleineres Modell
python train.py --config small
```

### "num_channels must be divisible by num_groups"
- Base Channels muss durch 32 teilbar sein (für GroupNorm)
- Verwenden Sie: 32, 64, 96, 128, 160, ...

### WandB Connection Error
```bash
# WandB Server neu starten
cd Imagenette
docker-compose restart
```

### Samples sind nur Rauschen
- Model ist nicht trainiert genug (> 20 Epochen trainieren)
- Verwenden Sie EMA Weights: `--use_ema`
- Guidance Scale anpassen: `--guidance_scale 5.0`

## 📚 Referenzen

- **DDPM Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- **Improved DDPM:** [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- **Classifier-Free Guidance:** [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)

## 🚀 Nächste Schritte (v2/v3)

**v2 - Improvements:**
- Höhere Auflösung (64x64 oder 128x128)
- DDIM Sampler (schnelleres Sampling)
- Bessere Conditioning Mechanismen

**v3 - Latent Diffusion (Stable Diffusion Style):**
- VAE (Variational Autoencoder) für Latent Space
- Text-Encoder (Transformer/BERT) für Text-zu-Bild
- Cross-Attention für Text Conditioning
- Training auf komplexeren Datasets

---

**Status:** ✅ Vollständig implementiert und getestet

Viel Erfolg beim Training! 🎉
