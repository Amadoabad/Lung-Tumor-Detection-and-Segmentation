import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deeplabv3 import visualize_segmentation_results,segment_with_boxes,debug_segmentation_crops,create_dataloaders,get_deeplabv3_model,CombinedLoss,train_model,evaluate_and_visualize,BCEDiceLoss,validate_model
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "Dataset/train"
val_dir = "Dataset/val"
batch_size = 8
checkpoint = torch.load('weights/full_seg_best_model.pth', map_location=torch.device('cpu'))
model = get_deeplabv3_model(num_classes=1)  # Binary segmentation
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)


# First, debug individual crops to see if predictions look correct
test_img_path = "Dataset/val/images/Subject_58/280.png"
test_bbox_path = "Dataset/val/detections/Subject_58/280.txt"
test_mask_image = 'Dataset/val/masks/Subject_58/280.png'
inference_transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])
debug_segmentation_crops(
    model=model,
    image_path=test_img_path,
    bbox_path=test_bbox_path,
    device=device,
    transform=inference_transform
)

# Then run the full segmentation
results = segment_with_boxes(
    model=model,
    image_path=test_img_path,
    bbox_path=test_bbox_path,
    device=device,
    transform=inference_transform,
    gt_mask_path=test_mask_image
)

visualize_segmentation_results(results)
