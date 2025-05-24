# DeepLabV3 for Object Segmentation - Two Architectures

This repository contains two architectures for object segmentation using DeepLabV3:

1. **Full-image segmentation**: Trained on complete images with masks
2. **Cropped-region segmentation**: Trained on individual object crops (works with YOLO detection)

## Architecture 1: Full-Image Segmentation

### Overview
- Processes entire images at once
- Outputs segmentation masks for all objects in the image
- Best for scenarios where object detection isn't available

### Dataset Structure
```
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── masks/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── detections/ (not used for this architecture)
```

### Training
```python
train_dataset = MultiObjectSameClassDatasetFixed(
    root_dir=train_dir,
    transform=train_transform,
    task="segmentation"
)
```

## Architecture 2: Cropped-Region Segmentation (YOLO-Compatible)

### Overview
- Processes individual object crops
- Designed to work with YOLO detection outputs
- Takes bounding boxes from YOLO and segments only those regions
- More efficient when combined with a detector

### Dataset Structure
```
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── masks/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── detections/ (YOLO format)
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Training
```python
train_dataset = BoxCroppedSegmentationDataset(
    root_dir=train_dir,
    transform=train_transform,
    target_format="yolo",
    max_boxes_per_image=3
)
```

## Key Features

### Common Components
- DeepLabV3 with ResNet50 backbone
- Combined BCE + Dice loss
- Albumentations for data augmentation
- Multi-GPU training support

### Cropped-Region Specific
- Bounding box aware cropping
- YOLO annotation compatibility
- Dynamic mask placement
- Multi-box processing (up to 3 boxes per image)

## Usage Examples

### Inference with Full-Image Model
```python
model = get_deeplabv3_model(num_classes=1)
image, mask = full_image_segment(model, "image.png", device, transform)
```

### Inference with YOLO + Cropped Model
```python
model = get_deeplabv3_model(num_classes=1)
image, mask, blended = segment_with_boxes(
    model, 
    "image.png", 
    "detections/image.txt", 
    device, 
    transform
)
```

## Training Parameters

```python
# Common parameters
batch_size = 8
num_epochs = 50
learning_rate = 0.001

# Loss weights
bce_weight = 0.5  # Balance between BCE and Dice loss

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

## Performance Notes

1. **Full-image model**:
   - Better for small objects
   - More computationally expensive
   - Requires precise masks

2. **Cropped-region model**:
   - Works with YOLO detections
   - More efficient (processes only relevant regions)
   - Better for deployment with existing detectors

## Directory Structure

```
project/
├── full_image_model/       # Architecture 1 files
├── cropped_region_model/   # Architecture 2 files
├── datasets/               # Training data
├── utils/                  # Common utilities
└── inference/              # Example inference scripts
```

## Dependencies

- PyTorch 1.8+
- Albumentations
- OpenCV
- NumPy
- tqdm
