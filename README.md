# Lung Tumor Detection and Segmentation

This project aims to develop a computer vision system capable of detecting and segmenting lung tumors in medical images, such as CT scans. The system leverages deep learning architectures for efficient and accurate tumor localization, which can support radiologists and healthcare professionals in diagnosis and treatment planning.

## Features

- **State-of-the-art Models**: Implements multiple segmentation models:
  - [U-Net](models/unet.py)
  - [U-Net++ (UNetPlusPlus)](models/unetpp.py)
  - [Attention U-Net](models/attention_unet.py)
- **Custom Loss Functions**: Supports advanced loss functions for better segmentation performance ([BCEDiceLoss, TverskyLoss](train/losses.py)).
- **Flexible Data Pipeline**: Modular data loaders and augmentation pipeline powered by [albumentations](https://albumentations.ai/) ([train/utils.py](train/utils.py)).
- **Visualization Tools**: Functions to visualize batches, segmentation overlays, and results.

## Directory Structure

```
.
├── data/                  # (gitignored) Place training and validation data here
├── models/                # Model architectures: unet.py, unetpp.py, attention_unet.py
├── train/                 # Training scripts, utilities, and loss functions
├── outputs/               # (gitignored) Model outputs and results
├── .gitignore
└── README.md
```

## Getting Started

### 1. Setup

Install required Python packages:

```bash
pip install torch torchvision albumentations matplotlib pillow
```

### 2. Prepare Dataset

- Organize your data into `data/train/` and `data/val/` directories.
- Each directory should contain pairs of images and corresponding masks.

### 3. Training

Sample code to create data loaders and train a model:

```python
from train.utils import get_dataloaders, get_transforms, get_optimizer, get_plateau_scheduler, init_weights_he
from models.unet import UNet
from train.losses import BCEDiceLoss

# Data loaders
train_loader, val_loader = get_dataloaders(
    train_dir='data/train',
    val_dir='data/val',
    train_batch_size=8,
    val_batch_size=4,
    transforms=get_transforms()
)

# Model
model = UNet(in_channels=1, out_channels=1)
model.apply(init_weights_he)

# Optimizer & scheduler
optimizer = get_optimizer(model, optimizer_name="adam", lr=1e-4)
scheduler = get_plateau_scheduler(optimizer)

# Loss
criterion = BCEDiceLoss()

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Validate, update scheduler, etc.
```

### 4. Visualization

Visualize predictions and overlays:

```python
from train.utils import visualize_batch, visualize_segmentation_overlay

# For a batch of images and masks
visualize_batch(images, masks)

# For a single image and mask
visualize_segmentation_overlay(original_image, predicted_mask)
```

## Models

### U-Net

- Classic encoder-decoder architecture with skip connections.
- See [models/unet.py](models/unet.py) for implementation.

### U-Net++

- Nested and dense skip pathways for improved feature fusion.
- See [models/unetpp.py](models/unetpp.py).

### Attention U-Net

- Incorporates attention gates to focus on relevant features.
- See [models/attention_unet.py](models/attention_unet.py).

## Loss Functions

- **BCEDiceLoss**: Combines Binary Cross-Entropy and Dice loss for robust binary segmentation.
- **TverskyLoss**: Variant of Dice loss to handle class imbalance.

See [train/losses.py](train/losses.py) for details and usage.

## Notes

- Temporary and output files are ignored via `.gitignore` (see [`.gitignore`](.gitignore) for details).
- The project is modular; you can plug in new models, loss functions, or augmentations as needed.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

---

For questions or contributions, please open an issue or submit a pull request!
