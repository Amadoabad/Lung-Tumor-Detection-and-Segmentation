import os
import torch
import numpy as np
import yaml
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import yolo
model_path= '../weights/best_small_object_detector.pt'
yolo_model = YOLO(model_path)
yaml_path = '../models/yaml/dataset.yaml'
results = yolo_model.train(
        device='cpu',
        resume=True,
        data=yaml_path,
        epochs=150,  # Increased from 100
        imgsz=640,   # Slightly larger than 600 (standard YOLO size)
        batch=16 if torch.cuda.is_available() else 4,  # Smaller batch for better gradient estimates

        # Enhanced optimizer settings
        optimizer='AdamW',
        lr0=0.01,   # Higher initial LR with better scheduling
        lrf=0.005,   # More gradual reduction
        warmup_epochs=5.0,

        # Adjusted loss weights for small objects
        box=8.0,     # Reduced from 7.5
        cls=.9,     # Increased from 0.5
        dfl=1.2,     # Slightly reduced

        # Enhanced augmentation
        hsv_h=0.01,  # Reduced hue variation
        hsv_s=0.5,   # Reduced saturation variation
        hsv_v=0.3,   # Reduced value variation
        translate=0.1,  # Reduced translation
        mosaic=0.8,  # Slightly reduced mosaic probability
        mixup=0.05,  # Reduced mixup
        copy_paste=0.05,  # Reduced copy-paste

        # Small object specific
        multi_scale=True,  # Try without multi-scale
        degrees=1.0,  # Reduced rotation
        perspective=0.0,  # No perspective change
        overlap_mask=True,

        # Additional parameters
        dropout=0.1,  # Add dropout for regularization
        patience=20,  # Longer patience
    )

