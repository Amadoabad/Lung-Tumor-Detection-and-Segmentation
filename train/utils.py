import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets.lung_dataset import LungDataset
from torch.utils.data import DataLoader

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
   
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


def get_dataloaders(train_dir, val_dir, batch_size, transforms, num_workers=4):
    train_dataset = LungDataset(train_dir, transforms=transforms)
    val_dataset = LungDataset(val_dir, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def visualize_batch(images, targets, format_type="YOLO"):
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
            # mask = target.permute(1, 2, 0).numpy()  # Assuming mask is a binary mask of shape [H, W]
            mask = target.numpy()  # Assuming mask is a binary mask of shape [H, W]
            axes[i].imshow(mask, alpha=0.5, cmap='gray')  # Overlay the mask with alpha transparency

        # Add a rectangle to the plot for detection tasks

    plt.tight_layout()
    plt.show()