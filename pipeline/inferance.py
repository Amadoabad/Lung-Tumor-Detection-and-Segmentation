import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import albumentations as A
from ultralytics import YOLO
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deeplabv3 import get_deeplabv3_model,segment_with_boxes,visualize_segmentation_results

model = get_deeplabv3_model(num_classes=1)
checkpoint = torch.load('weights/partila_seg_last_model.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

test_img_path = "Dataset/val/images/Subject_58/280.png"
test_bbox_path = "Dataset/val/images/Subject_58/280.txt"
test_mask_image = 'Dataset/val/masks/Subject_58/280.png'
inference_transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

model_path= 'weights/best_small_object_detector.pt'
yolo_model = YOLO(model_path)
yolo_results = yolo_model(test_img_path)

# Then perform segmentation
results = segment_with_boxes(
    model=model,
    image_path=test_img_path,
    yolo_results=yolo_results,
    device="cpu",
    transform=inference_transform,
    gt_mask_path=test_mask_image,  # optional
    draw_boxes=True  # Enable box drawing
)

visualize_segmentation_results(results,show_boxes=True)
