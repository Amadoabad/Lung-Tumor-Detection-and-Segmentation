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
from models.deeplabv3 import create_dataloaders,get_deeplabv3_model,CombinedLoss,train_model,evaluate_and_visualize,BCEDiceLoss,validate_model
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "Dataset/train"
val_dir = "Dataset/val"
batch_size = 8
checkpoint = torch.load('weights/full_seg_best_model.pth', map_location=torch.device('cpu'))
model = get_deeplabv3_model(num_classes=1)  # Binary segmentation
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
train_loader, val_loader = create_dataloaders(train_dir, val_dir, batch_size)

loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)
validate_model(val_loader, model, loss_fn,device)

figures = evaluate_and_visualize(model, val_loader, device)
for i, fig in enumerate(figures):
    fig.savefig(f'segmentation_result_{i}.png')
    plt.close(fig)
