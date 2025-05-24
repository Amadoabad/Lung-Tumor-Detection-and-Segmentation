# Lung Tumor Detection and Segmentation

A comprehensive deep learning pipeline for lung tumor detection and segmentation using YOLO for object detection and DeepLabV3 for precise segmentation.

## 🎯 Project Overview

This project implements a two-stage approach for lung tumor analysis:
1. **Detection Stage**: YOLO model detects tumor regions in lung images
2. **Segmentation Stage**: DeepLabV3 model performs precise pixel-level segmentation on detected regions

## 📁 Project Structure

```
LUNG-TUMOR-DETECTION-AND-SEGMENTATION/
├── Dataset/
│   ├── train/                    # Training images for YOLO
│   └── val/                      # Validation images for YOLO
├── datasets_class/
│   ├── __pycache__/
│   ├── __init__.py
│   └── lung_dataset.py          # Custom dataset classes
├── models/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── deeplabv3.py             # DeepLabV3 model implementation
│   └── yolo.py                  # YOLO model utilities
├── pipeline/
│   └── inference.py             # Main inference pipeline
├── playground/
│   └── Copy_of_CV_project.ipynb # Jupyter notebook for experiments
├── runs/
│   └── detect/                  # YOLO training outputs
│       ├── train5/
│       └── train52/
└── train_validation/
    ├── __pycache__/
    ├── __init__.py
    ├── predict_inference.py     # Prediction utilities
    ├── train.py                 # YOLO training script
    └── validation_test.py       # Validation and testing
└──weight/
    ├── best_small_object_detector.pt
    ├── partila_seg_last_model.pth
```

## 🚀 Features

- **Two-Stage Pipeline**: Combines detection and segmentation for accurate results
- **YOLO Detection**: Fast and accurate tumor detection
- **DeepLabV3 Segmentation**: Precise pixel-level tumor boundary delineation
- **Modular Design**: Separate training and inference components
- **Custom Dataset Support**: Flexible dataset handling for medical images
- **Comprehensive Evaluation**: Training, validation, and testing frameworks

## 🔧 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Ultralytics (for YOLO)
- OpenCV
- NumPy
- Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/lung-tumor-detection-and-segmentation.git
cd lung-tumor-detection-and-segmentation

# Install required packages
pip install torch torchvision ultralytics opencv-python matplotlib numpy albumentations

# Install additional dependencies
pip install -r requirements.txt  # if available
```

## 📊 Dataset

### YOLO Dataset Structure
```
Dataset/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO format annotations (.txt)
└── val/
    ├── images/          # Validation images
    └── labels/          # YOLO format annotations (.txt)
```

### Annotation Format
YOLO format: `class_id center_x center_y width height` (normalized coordinates)

## 🏋️ Training

### YOLO Training
```bash
# Train YOLO model
python train_validation/train.py

# Validate YOLO model
python train_validation/validation_test.py
```

### DeepLabV3 Training
> **Note**: DeepLabV3 training and validation code is available in a separate branch.
> 
> ```bash
> # Switch to DeepLabV3 training branch
> git checkout deeplabv3
> 
> # Follow training instructions in that branch
> ```

## 🔮 Inference

### Pipeline Usage
```bash

python /pipeline/inferance.py
```

## 📈 Model Performance

### YOLO Detection Metrics
- **mAP@0.5**: 0.6
- **Precision**: 0.9
- **Recall**: 0.5

### DeepLabV3 Segmentation Metrics
- **Dice Score**: 72%
- **Pixel Accuracy**: 99%

## 🔄 Pipeline Workflow

1. **Input**: Lung CT/X-ray image
2. **YOLO Detection**: Identifies potential tumor regions
3. **ROI Extraction**: Crops detected regions
4. **DeepLabV3 Segmentation**: Performs pixel-level segmentation on cropped regions
5. **Result Fusion**: Combines segmentation masks back to original image coordinates
6. **Output**: Annotated image with detected and segmented tumors

## 📁 Key Files

- `train_validation/train.py`: YOLO model training
- `train_validation/validation_test.py`: Model validation and testing
- `models/yolo.py`: YOLO model utilities and inference
- `models/deeplabv3.py`: DeepLabV3 model implementation
- `pipeline/inference.py`: Complete detection + segmentation pipeline
- `datasets_class/lung_dataset.py`: Custom dataset classes for data loading

## 🎨 Visualization

The pipeline provides comprehensive visualizations:
- Original images with detection bounding boxes
- Segmentation masks overlaid on original images
- Comparison between ground truth and predictions
- Performance metrics and evaluation plots
