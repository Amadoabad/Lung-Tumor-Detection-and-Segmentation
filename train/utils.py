import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from datasets.lung_dataset import LungDataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import numpy as np
from pathlib import Path


def get_dataloaders(train_dir, val_dir, train_batch_size, val_batch_size=None, transforms=None, num_workers=4):
    """    
        Creates PyTorch DataLoader objects for training and validation datasets.
        
        train_dir (str or Path): Path to the training data directory.
        val_dir (str or Path): Path to the validation data directory.
        train_batch_size (int): Batch size for the training DataLoader.
        val_batch_size (int, optional): Batch size for the validation DataLoader. If None, uses train_batch_size. Defaults to None.
        transforms (callable, optional): Transformations to apply to the datasets. Defaults to None.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
        
        tuple: (train_loader, val_loader)
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
    """

    if val_batch_size is None:
        val_batch_size = train_batch_size
    
    train_dataset = LungDataset(train_dir, transforms=transforms)
    val_dataset = LungDataset(val_dir, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_transforms():
    """
    Creates a set of transformations to be applied to the images and masks.
    The transformations include:
        - Normalization: Normalizes the image to have mean=0 and std=1.
        - ToTensorV2: Converts the image and mask to PyTorch tensors.
    Returns:
        A.Compose: A composed transformation object that applies the transformations in sequence.
    
    """
    transform = A.Compose([
    A.Normalize(
        mean=0.0,
        std=1.0,
        max_pixel_value=255.0
    ),
    ToTensorV2()       # Convert to PyTorch tensor
    ])
    return transform

def read_image(image_path):
    """
    Reads an image from the given path and converts it to grayscale.
    
    Args:
        image_path (str or Path): Path to the image file.
        
    Returns:
        PIL Image: The loaded image in grayscale mode.
    """
    image = Image.open(image_path).convert('L')
    return image

def preprocess_image(image):
    """
    Preprocesses the input image and normalizing it.
    
    Args:
        image (PIL Image or numpy array): The input image to be preprocessed.
        
    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor.
    """
    image = np.array(image)
    image = get_transforms()(image=image)['image'] 
    image = image.unsqueeze(0)  # Add batch dimension

    return image

def init_weights_he(m):
    """
    Initialize weights using He initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
        
def get_optimizer(model, optimizer_name="adam", lr=1e-4, weight_decay=0):
    """
    Creates an optimizer for the given model.
    Args:
        model (torch.nn.Module): The model for which to create the optimizer.
        optimizer_name (str, optional): The name of the optimizer to use. Default is "adam".
        lr (float, optional): Learning rate. Default is 1e-4.
        weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
    Returns:
        torch.optim.Optimizer: The created optimizer.
    Raises:
        ValueError: If the optimizer name is not supported.
    Notes:
        - Currently supports "adam" and "sgd" optimizers.
        - The learning rate and weight decay can be adjusted as needed.
        - The optimizer is created with the model's parameters.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_plateau_scheduler(optimizer, mode='min', patience=2, factor=0.1):
    """
    Creates a ReduceLROnPlateau scheduler for the given optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to attach the scheduler to.
        mode (str, optional): One of 'min', 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing. Default is 'min'.
        patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Default is 2.
        factor (float, optional): Factor by which the learning rate will be reduced. new_lr = lr * factor. Default is 0.1.
    
    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: The created scheduler.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    

def visualize_batch(images, targets, format_type="Segmentation"):
    """
    Visualizes a batch of images along with their corresponding targets (e.g., segmentation masks).
    Args:
        images (list or tensor): A list or batch of image tensors, each of shape [C, H, W].
        targets (list or tensor): A list or batch of target tensors (e.g., segmentation masks), each of shape [H, W].
        format_type (str, optional): The type of visualization to perform. 
            If "Segmentation", overlays the target mask on the image. Default is "Segmentation".
    Notes:
        - Assumes images are PyTorch tensors and are in [C, H, W] format.
        - Assumes targets are PyTorch tensors and are in [H, W] format for segmentation.
        - Displays the images and overlays the masks with alpha transparency for segmentation tasks.
    """
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    if len(images) == 1:
        axes = [axes]  # Ensure axes is iterable for a single image

    for i, (img, target) in enumerate(zip(images, targets)):
        # Convert image tensor to NumPy array
        img_np = img.permute(1, 2, 0).numpy()

        # Plot the image
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis("off")

        if format_type == "Segmentation":
            # For segmentation, plot the mask
            mask = target.numpy()  # Assuming mask is a binary mask of shape [H, W]
            axes[i].imshow(mask, alpha=0.5, cmap='gray')  # Overlay the mask with alpha transparency


    plt.tight_layout()
    plt.show()