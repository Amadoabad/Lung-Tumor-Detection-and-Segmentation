import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_and_load_model(model_path, device='cuda'):
    """
    Simple function to create model architecture and load weights
    """
    print("Creating model architecture...")

    # Create the model architecture (same as training)
    model = maskrcnn_resnet50_fpn(pretrained=False)

    # Replace the classifier head for bounding box prediction
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 classes: background + tumor

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

    print("Loading saved weights...")

    # Load the saved weights
    checkpoint = torch.load(model_path, map_location=device)

    # Load state dict (handle different save formats)
    if isinstance(checkpoint, dict):
        # Check for common keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully on {device}")
    print(f"Model type: {type(model)}")

    return model




def visualize_single_image_prediction(model_path ,image_path, score_threshold=0.3):
    """
    Visualize Mask R-CNN prediction for a single image loaded from a file path.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_and_load_model(model_path, device=device)


    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Apply same transforms as your dataset (assumed normalized to [0,1])
    transform = T.Compose([
        T.ToTensor(),  # Converts to tensor and normalizes to [0,1]
    ])
    image_tensor = transform(image).to(device)

    # Keep a version for display
    original_image = image_tensor.permute(1, 2, 0).cpu().numpy()
    original_image = np.clip(original_image, 0, 1)

    # Model prediction
    with torch.no_grad():
        prediction = model([image_tensor])

    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(original_image)
    ax.set_title(f'Predictions (threshold: {score_threshold})')
    ax.axis('off')

    # Get predictions
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_masks = prediction[0]['masks'].cpu().numpy()

    valid_indices = pred_scores > score_threshold
    if np.any(valid_indices):
        filtered_boxes = pred_boxes[valid_indices]
        filtered_scores = pred_scores[valid_indices]
        filtered_masks = pred_masks[valid_indices]

        # Draw masks
        for j, (mask, score) in enumerate(zip(filtered_masks, filtered_scores)):
            mask_np = mask.squeeze()
            mask_binary = mask_np > 0.5
            color_idx = j % 10
            colors = plt.cm.Set3(color_idx)[:3]
            colored_mask = np.zeros((*mask_binary.shape, 3))
            colored_mask[mask_binary] = colors
            ax.imshow(colored_mask, alpha=0.5)

        # Draw boxes
        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score:.2f}",
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=10, color='red')
    else:
        ax.text(0.5, 0.5, 'No detections above threshold',
                transform=ax.transAxes,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    model_path = 'weight/best_tumor_segmentation_model.pth'  # Path to your saved model
    image_path = 'data/val/images/Subject_60/48.png'  # Path to your test image
    visualize_single_image_prediction(model_path, image_path, score_threshold=0.9)