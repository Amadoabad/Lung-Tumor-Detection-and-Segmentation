import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.lung_dataset import TumorSegmentationDataset, collate_fn, get_transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
batch_size = 16  # Small batch size due to memory constraints
num_epochs = 50
learning_rate = 0.00001

# Create datasets
train_transform = get_transforms()
val_transform = get_transforms()

# Update these paths to your dataset
train_dataset = TumorSegmentationDataset('data/train', transform=train_transform, mode='train')
val_dataset = TumorSegmentationDataset('data/val', transform=val_transform, mode='val')

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


