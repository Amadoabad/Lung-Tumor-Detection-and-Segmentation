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
from models.deeplabv3 import create_dataloaders,get_deeplabv3_model,CombinedLoss,train_model
# Paths to dataset directories
train_dir = "Dataset/train"
val_dir = "Dataset/val"

# Hyperparameters
batch_size = 8
num_epochs = 20
learning_rate = 3e-4
weight_decay = 1e-4

# Create dataloaders
train_loader, val_loader = create_dataloaders(train_dir, val_dir, batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = get_deeplabv3_model(num_classes=1)  # Binary segmentation
model = model.to(device)

# Define loss function and optimizer
criterion = CombinedLoss(weight=0.7)  # Weighted combination of BCE and Dice Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    scheduler=scheduler
)
print("Training completed!")
