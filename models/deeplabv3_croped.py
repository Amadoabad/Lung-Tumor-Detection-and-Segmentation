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
def segment_with_boxes(model, image_path, bbox_path, device, transform, gt_mask_path=None):
    """
    Performs segmentation on cropped regions defined by bounding boxes in format: x1,y1,x2,y2
    and places the segmentation results back onto the original image. Optionally compares with ground truth.

    Args:
        model: Trained segmentation model
        image_path: Path to the input image
        bbox_path: Path to the bounding box file (format: x1,y1,x2,y2 per line)
        device: Device to run inference on
        transform: Image transformations
        gt_mask_path: Path to ground truth segmentation mask image (optional)

    Returns:
        Dictionary containing:
        - 'original': Original image
        - 'pred_mask': Predicted segmentation mask
        - 'pred_overlay': Prediction overlaid on original image
        - 'gt_mask': Ground truth mask (if provided)
        - 'gt_overlay': Ground truth overlaid on original image (if provided)
        - 'comparison_overlay': Both masks overlaid on original image (if GT provided)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Create empty mask for the full image
    full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path is not None:
        try:
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Resize ground truth mask to match original image if needed
                if gt_mask.shape != (orig_h, orig_w):
                    gt_mask = cv2.resize(gt_mask, (orig_w, orig_h))
                # Normalize to 0-1
                gt_mask = (gt_mask > 128).astype(np.float32)
            else:
                print(f"Warning: Could not load ground truth mask: {gt_mask_path}")
        except Exception as e:
            print(f"Error loading ground truth mask: {e}")
            gt_mask = None

    # Read bounding boxes in x1,y1,x2,y2 format
    boxes = []

    try:
        with open(bbox_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:  # Format: x1,y1,x2,y2
                try:
                    x1, y1, x2, y2 = map(float, parts[:4])
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    boxes.append([x1, y1, x2, y2])
                except ValueError:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue

    except FileNotFoundError:
        print(f"Bounding box file not found: {bbox_path}")
        results = {
            'original': image,
            'pred_mask': full_mask,
            'pred_overlay': image.copy()
        }
        if gt_mask is not None:
            results['gt_mask'] = gt_mask
            results['gt_overlay'] = image.copy()
            results['comparison_overlay'] = image.copy()
        return results
    except Exception as e:
        print(f"Failed to parse bounding box file: {bbox_path}, Error: {e}")
        results = {
            'original': image,
            'pred_mask': full_mask,
            'pred_overlay': image.copy()
        }
        if gt_mask is not None:
            results['gt_mask'] = gt_mask
            results['gt_overlay'] = image.copy()
            results['comparison_overlay'] = image.copy()
        return results

    model.eval()

    # If no boxes found, return the original image
    if not boxes:
        print(f"No valid bounding boxes found in {bbox_path}")
        results = {
            'original': image,
            'pred_mask': full_mask,
            'pred_overlay': image.copy()
        }
        if gt_mask is not None:
            results['gt_mask'] = gt_mask
            results['gt_overlay'] = image.copy()
            results['comparison_overlay'] = image.copy()
        return results

    # Process each bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Skip if box is too small or invalid
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
            print(f"Warning: Skipping box {i+1} - invalid or too small ({x2-x1}x{y2-y1})")
            continue

        print(f"Processing box {i+1}: ({x1},{y1}) to ({x2},{y2}) - size: {x2-x1}x{y2-y1}")

        # Crop image - ensure we get the exact region
        cropped_image = image[y1:y2, x1:x2].copy()

        if cropped_image.size == 0:
            print(f"Warning: Empty crop for box {i+1}")
            continue

        # Apply transformations
        try:
            augmented = transform(image=cropped_image)
            cropped_tensor = augmented['image'].float().unsqueeze(0).to(device)
        except Exception as e:
            print(f"Warning: Transform failed for box {i+1}: {e}")
            continue

        # Get prediction
        with torch.no_grad():
            output = model(cropped_tensor)['out']
            pred_mask = torch.sigmoid(output) > 0.5
            pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.float32)

        # Get the actual dimensions of the cropped region
        crop_h, crop_w = y2 - y1, x2 - x1

        # Resize mask back to exact crop size using proper interpolation
        if pred_mask.shape != (crop_h, crop_w):
            pred_mask_resized = cv2.resize(pred_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask_resized = pred_mask

        # Ensure the resized mask fits exactly in the crop region
        pred_mask_resized = pred_mask_resized.astype(np.float32)

        # Place the mask in the full image - use maximum to handle overlapping regions
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], pred_mask_resized)

    # Create overlays
    # Prediction overlay (red)
    pred_colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    pred_colored_mask[full_mask == 1] = [255, 0, 0]  # Red overlay for predictions
    pred_overlay = cv2.addWeighted(image, 1, pred_colored_mask, 0.5, 0)

    # Prepare results dictionary
    results = {
        'original': image,
        'pred_mask': full_mask,
        'pred_overlay': pred_overlay
    }

    # If ground truth is available, create additional overlays
    if gt_mask is not None:
        # Ground truth overlay (green)
        gt_colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        gt_colored_mask[gt_mask == 1] = [0, 255, 0]  # Green overlay for ground truth
        gt_overlay = cv2.addWeighted(image, 1, gt_colored_mask, 0.5, 0)

        # Comparison overlay (GT in green, Pred in red, overlap in yellow)
        comparison_colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        comparison_colored_mask[gt_mask == 1] = [0, 255, 0]  # Green for ground truth

        # Add red for predictions (cast to int32 to avoid overflow, then clip)
        red_mask = np.zeros_like(comparison_colored_mask, dtype=np.int32)
        red_mask[full_mask == 1] = [255, 0, 0]
        comparison_colored_mask = comparison_colored_mask.astype(np.int32) + red_mask
        comparison_colored_mask = np.clip(comparison_colored_mask, 0, 255).astype(np.uint8)
        # Where both are present, it will be yellow (green + red)
        comparison_overlay = cv2.addWeighted(image, 1, comparison_colored_mask, 0.4, 0)

        results.update({
            'gt_mask': gt_mask,
            'gt_overlay': gt_overlay,
            'comparison_overlay': comparison_overlay
        })

    return results


def debug_segmentation_crops(model, image_path, bbox_path, device, transform, gt_mask_path=None):
    """
    Debug version that shows individual crops and their predictions.
    Fixed to handle shape mismatches between predicted masks and cropped images.
    """
    import matplotlib.pyplot as plt

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Read bounding boxes in x1,y1,x2,y2 format
    boxes = []
    try:
        with open(bbox_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    x1, y1, x2, y2 = map(float, parts[:4])
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    boxes.append([x1, y1, x2, y2])
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading boxes: {e}")
        return

    model.eval()

    # Show each crop and its prediction
    for i, box in enumerate(boxes[:6]):  # Limit to first 6 boxes
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop image
        cropped_image = image[y1:y2, x1:x2].copy()

        # Get prediction
        augmented = transform(image=cropped_image)
        cropped_tensor = augmented['image'].float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(cropped_tensor)['out']
            pred_mask = torch.sigmoid(output) > 0.5
            pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.float32)

        # FIXED: Resize prediction mask to match cropped image dimensions
        crop_h, crop_w = cropped_image.shape[:2]
        if pred_mask.shape != (crop_h, crop_w):
            pred_mask_resized = cv2.resize(pred_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask_resized = pred_mask

        # Visualize
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cropped_image)
        plt.title(f"Crop {i+1}: {cropped_image.shape[:2]}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask_resized, cmap='gray')
        plt.title(f"Prediction: {pred_mask_resized.shape}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cropped_image)
        # FIXED: Create overlay with matching dimensions
        overlay = np.zeros_like(cropped_image)
        overlay[:, :, 0] = pred_mask_resized * 255  # Now shapes match
        plt.imshow(overlay, alpha=0.5)
        plt.title("Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Box {i+1}: Original crop size: {cropped_image.shape[:2]}, "
              f"Prediction size: {pred_mask.shape}, "
              f"Resized prediction: {pred_mask_resized.shape}")


def visualize_segmentation_results(results, figsize=(20, 5)):
    """
    Visualize the segmentation results from segment_with_boxes function.

    Args:
        results: Dictionary returned by segment_with_boxes function
        figsize: Figure size for matplotlib
    """
    import matplotlib.pyplot as plt

    # Determine number of subplots based on available data
    has_gt = 'gt_mask' in results
    num_plots = 6 if has_gt else 3

    plt.figure(figsize=figsize)

    # Original image
    plt.subplot(2 if has_gt else 1, 3, 1)
    plt.imshow(results['original'])
    plt.title("Original Image")
    plt.axis('off')

    # Predicted mask
    plt.subplot(2 if has_gt else 1, 3, 2)
    plt.imshow(results['pred_mask'], cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    # Prediction overlay
    plt.subplot(2 if has_gt else 1, 3, 3)
    plt.imshow(results['pred_overlay'])
    plt.title("Prediction Overlay (Red)")
    plt.axis('off')

    if has_gt:
        # Ground truth mask
        plt.subplot(2, 3, 4)
        plt.imshow(results['gt_mask'], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Ground truth overlay
        plt.subplot(2, 3, 5)
        plt.imshow(results['gt_overlay'])
        plt.title("Ground Truth Overlay (Green)")
        plt.axis('off')

        # Comparison overlay
        plt.subplot(2, 3, 6)
        plt.imshow(results['comparison_overlay'])
        plt.title("Comparison: GT(Green) + Pred(Red)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

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
