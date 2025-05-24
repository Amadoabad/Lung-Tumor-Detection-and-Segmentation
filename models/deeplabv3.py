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
from datasets.lung_dataset import MultiObjectSameClassDatasetFixed,BoxCroppedSegmentationDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DeepLabV3 model
def get_deeplabv3_model(num_classes=1):
    # Initialize model with pretrained weights
    model = deeplabv3_resnet50(pretrained=True)

    # Modify the classifier for binary segmentation (foreground/background)
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1,
        stride=1
    )

    return model





def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, weight=0.5):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = dice_loss(pred, target)

        return bce * self.weight + dice * (1 - self.weight)

# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(dataloader):
        # Move data to device
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)  # Add channel dimension and convert to float

        # Forward pass
        outputs = model(images)['out']

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            # Move data to device
            images = images.to(device)
            masks = masks.float().unsqueeze(1).to(device)  # Add channel dimension and convert to float

            # Forward pass
            outputs = model(images)['out']

            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluate model and visualize cropped images and their masks
def evaluate_and_visualize(model, dataloader, device, num_samples=5):
    model.eval()

    # Get some random images from validation set
    dataiter = iter(dataloader)
    figures = []

    for i in range(min(num_samples, len(dataloader))):
        images, masks = next(dataiter)

        # Move to device
        images = images.to(device)
        masks = masks.to(device)

        # Get predictions
        with torch.no_grad():
            preds = model(images)['out']
            preds = torch.sigmoid(preds) > 0.5

        # Move back to CPU for visualization
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

        # Convert from tensor format
        images = np.transpose(images, (0, 2, 3, 1))
        # Denormalize images
        images = images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        images = np.clip(images, 0, 1)

        # Plot the results
        for j in range(min(3, len(images))):
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(images[j])
            plt.title("Cropped Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(masks[j], cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(preds[j, 0], cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')

            plt.tight_layout()
            figures.append(plt)

    return figures

# Main training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, save_dir='checkpoints'):
    # Create directory to save checkpoints
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print("Saved best model checkpoint!")

        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(save_dir, 'last_model.pth'))

    return model, history

# Inference function to process full images with bounding boxes
def segment_with_boxes(model, image_path, bbox_path, device, transform):
    """
    Performs segmentation on cropped regions defined by bounding boxes in YOLO format
    and places the segmentation results back onto the original image.

    Args:
        model: Trained segmentation model
        image_path: Path to the input image
        bbox_path: Path to the bounding box file (in yolo format)
        device: Device to run inference on
        transform: Image transformations

    Returns:
        Original image and segmentation mask overlaid on the original image
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Create empty mask for the full image
    full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

    # Read bounding boxes in the annotation format used by MultiObjectSameClassDatasetFixed
    boxes = []

    try:
        # First try the YOLO format
        with open(bbox_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # YOLO format: class_id, x_center, y_center, width, height
                _, x_center, y_center, width, height = map(float, parts[:5])

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width/2) * orig_w)
                y1 = int((y_center - height/2) * orig_h)
                x2 = int((x_center + width/2) * orig_w)
                y2 = int((y_center + height/2) * orig_h)

                boxes.append([x1, y1, x2, y2])
    except:
        # Try the comma-separated format if YOLO format parsing fails
        try:
            with open(bbox_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                try:
                    data = list(map(int, line.strip().split(",")))
                    if len(data) == 4:  # x_min, y_min, x_max, y_max
                        boxes.append(data)
                except ValueError:
                    continue
        except:
            print(f"Failed to parse bounding box file: {bbox_path}")
            return image, full_mask, image  # Return original image if parsing fails

    model.eval()

    # If no boxes found, return the original image
    if not boxes:
        print(f"No valid bounding boxes found in {bbox_path}")
        return image, full_mask, image

    # Process each bounding box
    for box in boxes:
        x1, y1, x2, y2 = box

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Skip if box is too small
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        # Crop image
        cropped_image = image[y1:y2, x1:x2]

        # Apply transformations
        augmented = transform(image=cropped_image)
        cropped_tensor = augmented['image'].unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(cropped_tensor)['out']
            pred_mask = torch.sigmoid(output) > 0.5
            pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

        # Resize mask back to crop size
        pred_mask_resized = cv2.resize(pred_mask, (x2 - x1, y2 - y1))

        # Place the mask in the full image
        full_mask[y1:y2, x1:x2] = pred_mask_resized

    # Create colored overlay
    colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    colored_mask[full_mask == 1] = [0, 255, 0]  # Green overlay

    # Blend original image with mask
    alpha = 0.5
    blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    return image, full_mask, blended


def validate_model(loader, model, loss_fn, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.float().to(device).unsqueeze(1)
            

            logits = model(images)
            if isinstance(logits, dict):
                logits = logits['out'] 
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            num_correct += (preds == masks).sum().item()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * masks).sum().item()) / (
                (preds + masks).sum().item() + 1e-8
            )

    avg_loss = total_loss / len(loader)
    avg_dice = dice_score / len(loader)
    acc = num_correct / num_pixels * 100

    print(f"[Validation] Acc: {acc:.2f}%, Dice: {avg_dice:.4f}, Loss: {avg_loss:.4f}")
    return avg_loss, avg_dice


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def create_dataloaders(train_dir, val_dir, batch_size):
    # Create datasets
    train_dataset = MultiObjectSameClassDatasetFixed(
        root_dir=train_dir,
        transform=train_transform,
        target_format="yolo",  # Not really used for segmentation
        task="segmentation"
    )

    val_dataset = MultiObjectSameClassDatasetFixed(
        root_dir=val_dir,
        transform=val_transform,
        target_format="yolo",  # Not really used for segmentation
        task="segmentation"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def create_dataloaders_crop(train_dir, val_dir, batch_size):
    # Create datasets using the BoxCroppedSegmentationDataset
    train_dataset = BoxCroppedSegmentationDataset(
        root_dir=train_dir,
        transform=train_transform,
        target_format="yolo",
        max_boxes_per_image=3  # Use 1 box per image for training
    )

    val_dataset = BoxCroppedSegmentationDataset(
        root_dir=val_dir,
        transform=val_transform,
        target_format="yolo",
        max_boxes_per_image=3  # Use 1 box per image for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
