import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
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

def segment_with_boxes(model, image_path, yolo_results, device, transform, gt_mask_path=None, draw_boxes=True):
    """
    Performs segmentation on cropped regions defined by YOLO detection results
    and places the segmentation results back onto the original image. Optionally compares with ground truth.

    Args:
        model: Trained segmentation model
        image_path: Path to the input image
        yolo_results: YOLO detection results object (from model(image_path))
        device: Device to run inference on
        transform: Image transformations
        gt_mask_path: Path to ground truth segmentation mask image (optional)
        draw_boxes: Whether to draw YOLO bounding boxes on the images (default: True)

    Returns:
        Dictionary containing:
        - 'original': Original image
        - 'original_with_boxes': Original image with YOLO boxes drawn (if draw_boxes=True)
        - 'pred_mask': Predicted segmentation mask
        - 'pred_overlay': Prediction overlaid on original image
        - 'pred_overlay_with_boxes': Prediction overlay with YOLO boxes (if draw_boxes=True)
        - 'gt_mask': Ground truth mask (if provided)
        - 'gt_overlay': Ground truth overlaid on original image (if provided)
        - 'comparison_overlay': Both masks overlaid on original image (if GT provided)
        - 'comparison_overlay_with_boxes': Comparison overlay with YOLO boxes (if draw_boxes=True and GT provided)
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

    # Extract bounding boxes from YOLO results
    boxes = []
    confidences = []
    class_ids = []

    if hasattr(yolo_results[0], 'boxes'):
        for box in yolo_results[0].boxes:
            # Convert from xywh to xyxy if needed (YOLO typically uses xywh)
            if len(box.xyxy) > 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

                # Get confidence and class info if available
                if hasattr(box, 'conf') and len(box.conf) > 0:
                    confidences.append(float(box.conf[0].cpu().numpy()))
                else:
                    confidences.append(1.0)

                if hasattr(box, 'cls') and len(box.cls) > 0:
                    class_ids.append(int(box.cls[0].cpu().numpy()))
                else:
                    class_ids.append(0)

    def draw_yolo_boxes(img, boxes, confidences, class_ids, class_names=None):
        """Draw YOLO bounding boxes on image"""
        img_with_boxes = img.copy()

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box

            # Choose color based on class_id (cycle through colors)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            color = colors[cls_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            if class_names and cls_id < len(class_names):
                label = f"{class_names[cls_id]}: {conf:.2f}"
            else:
                label = f"Class {cls_id}: {conf:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw background rectangle for text
            cv2.rectangle(img_with_boxes,
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)

            # Draw text
            cv2.putText(img_with_boxes, label, (x1, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw box number
            cv2.putText(img_with_boxes, f"#{i+1}", (x1+2, y2-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return img_with_boxes

    model.eval()

    # If no boxes found, return the original image
    if not boxes:
        print("No valid bounding boxes found in YOLO results")
        results = {
            'original': image,
            'pred_mask': full_mask,
            'pred_overlay': image.copy()
        }
        if draw_boxes:
            results['original_with_boxes'] = image.copy()
            results['pred_overlay_with_boxes'] = image.copy()
        if gt_mask is not None:
            results['gt_mask'] = gt_mask
            results['gt_overlay'] = image.copy()
            results['comparison_overlay'] = image.copy()
            if draw_boxes:
                results['comparison_overlay_with_boxes'] = image.copy()
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

    # Get class names from YOLO results if available
    class_names = None
    if hasattr(yolo_results[0], 'names'):
        class_names = yolo_results[0].names

    # Prepare results dictionary
    results = {
        'original': image,
        'pred_mask': full_mask,
        'pred_overlay': pred_overlay
    }

    # Add images with bounding boxes if requested
    if draw_boxes:
        results['original_with_boxes'] = draw_yolo_boxes(image, boxes, confidences, class_ids, class_names)
        results['pred_overlay_with_boxes'] = draw_yolo_boxes(pred_overlay, boxes, confidences, class_ids, class_names)

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

        # Add comparison overlay with boxes if requested
        if draw_boxes:
            results['comparison_overlay_with_boxes'] = draw_yolo_boxes(
                comparison_overlay, boxes, confidences, class_ids, class_names
            )

    return results

def visualize_segmentation_results(results, figsize=(20, 12), show_boxes=True):
    """
    Visualizes segmentation results with optional ground truth comparison and YOLO boxes

    Args:
        results: Dictionary returned by segment_with_boxes()
        figsize: Size of the matplotlib figure
        show_boxes: Whether to show versions with YOLO boxes drawn (default: True)
    """
    # Determine if we have ground truth and boxes
    has_gt = 'gt_mask' in results
    has_boxes = show_boxes and 'original_with_boxes' in results

    # Calculate number of rows and columns
    if has_boxes:
        rows = 2
        cols = 4 if has_gt else 3
        fig_height = figsize[1]
    else:
        rows = 1
        cols = 4 if has_gt else 2
        fig_height = figsize[1] // 2

    plt.figure(figsize=(figsize[0], fig_height))

    plot_idx = 1

    # First row - original images
    plt.subplot(rows, cols, plot_idx)
    plt.imshow(results['original'])
    plt.title('Original Image')
    plt.axis('off')
    plot_idx += 1

    plt.subplot(rows, cols, plot_idx)
    plt.imshow(results['pred_overlay'])
    plt.title('Prediction (Red)')
    plt.axis('off')
    plot_idx += 1

    if has_gt:
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(results['gt_overlay'])
        plt.title('Ground Truth (Green)')
        plt.axis('off')
        plot_idx += 1

        plt.subplot(rows, cols, plot_idx)
        plt.imshow(results['comparison_overlay'])
        plt.title('Comparison (Yellow=Overlap)')
        plt.axis('off')
        plot_idx += 1

    # Second row - images with YOLO boxes
    if has_boxes:
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(results['original_with_boxes'])
        plt.title('Original + YOLO Boxes')
        plt.axis('off')
        plot_idx += 1

        plt.subplot(rows, cols, plot_idx)
        plt.imshow(results['pred_overlay_with_boxes'])
        plt.title('Prediction + YOLO Boxes')
        plt.axis('off')
        plot_idx += 1

        if has_gt:
            # For GT with boxes, we can use the original with boxes since GT doesn't need its own box version
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(results['original_with_boxes'])
            plt.title('Ground Truth + YOLO Boxes')
            plt.axis('off')
            plot_idx += 1

            plt.subplot(rows, cols, plot_idx)
            plt.imshow(results['comparison_overlay_with_boxes'])
            plt.title('Comparison + YOLO Boxes')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Additionally show the raw masks
    plt.figure(figsize=(figsize[0]//2, figsize[1]//3))

    # Plot prediction mask
    plt.subplot(1, 2 if has_gt else 1, 1)
    plt.imshow(results['pred_mask'], cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')

    if has_gt:
        plt.subplot(1, 2, 2)
        plt.imshow(results['gt_mask'], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Usage example:
"""
# Load your models
yolo_model = YOLO('path/to/yolo/model.pt')
segmentation_model = load_your_segmentation_model()

# Run YOLO detection
yolo_results = yolo_model('path/to/image.jpg')

# Run segmentation with box drawing
results = segment_with_boxes(
    model=segmentation_model,
    image_path='path/to/image.jpg',
    yolo_results=yolo_results,
    device=device,
    transform=your_transform,
    gt_mask_path='path/to/ground_truth.png',  # optional
    draw_boxes=True  # Enable box drawing
)

# Visualize results
visualize_segmentation_results(results, show_boxes=True)
"""
