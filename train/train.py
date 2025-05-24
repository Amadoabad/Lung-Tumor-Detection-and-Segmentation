from tqdm import tqdm
import torch

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0

    # Create progress bar for the entire epoch
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}', ncols=100, leave=True)

    for i, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Update progress bar with current loss
        pbar.set_postfix({'Loss': f'{losses.item():.4f}'})

    return total_loss / len(data_loader)




def evaluate(model, data_loader, device, epoch):
    """Evaluate the model with proper handling of empty predictions"""
    model.eval()
    total_loss = 0
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    total_gt_objects = 0  # Total ground truth objects
    total_pred_objects = 0  # Total predicted objects
    num_samples = 0

    # Create progress bar for validation
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} - Val', ncols=140, leave=True)

    with torch.no_grad():
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calculate validation loss (set model to train mode temporarily for loss calculation)
            model.train()
            loss_dict = model(images, targets)
            val_loss = sum(loss for loss in loss_dict.values())
            total_loss += val_loss.item()
            model.eval()

            # Get predictions for metrics calculation
            predictions = model(images)

            # Calculate metrics for each prediction
            batch_tp = 0
            batch_fp = 0
            batch_fn = 0
            batch_gt_objects = 0
            batch_pred_objects = 0

            for pred, target in zip(predictions, targets):
                # Count ground truth objects
                gt_count = len(target['masks'])
                batch_gt_objects += gt_count

                # Count predicted objects (with confidence > 0.5)
                if len(pred['masks']) > 0:
                    high_conf_mask = pred['scores'] > 0.5
                    pred_count = high_conf_mask.sum().item()
                    batch_pred_objects += pred_count

                    if pred_count > 0 and gt_count > 0:
                        # Get high confidence predictions
                        high_conf_indices = torch.where(high_conf_mask)[0]

                        # For simplicity, we'll use IoU-based matching
                        # Get the best prediction (highest score among high confidence ones)
                        if len(high_conf_indices) > 0:
                            best_idx = high_conf_indices[torch.argmax(pred['scores'][high_conf_indices])]
                            pred_mask = pred['masks'][best_idx, 0] > 0.5

                            # Compare with first ground truth mask
                            gt_mask = target['masks'][0] > 0

                            # Calculate IoU for matching
                            intersection = (pred_mask & gt_mask).sum().float()
                            union = (pred_mask | gt_mask).sum().float()
                            iou = intersection / union if union > 0 else 0

                            # Consider it a true positive if IoU > 0.5
                            if iou > 0.5:
                                batch_tp += 1
                            else:
                                batch_fp += 1
                                batch_fn += 1  # Also count as missed ground truth
                        else:
                            batch_fn += gt_count  # All ground truths are missed
                    elif pred_count > 0:
                        # Predictions but no ground truth
                        batch_fp += pred_count
                    elif gt_count > 0:
                        # Ground truth but no predictions
                        batch_fn += gt_count
                else:
                    # No predictions at all
                    batch_fn += gt_count

            total_tp += batch_tp
            total_fp += batch_fp
            total_fn += batch_fn
            total_gt_objects += batch_gt_objects
            total_pred_objects += batch_pred_objects
            num_samples += len(images)

            # Calculate batch metrics for display
            batch_precision = batch_tp / (batch_tp + batch_fp) if (batch_tp + batch_fp) > 0 else 0
            batch_recall = batch_tp / (batch_tp + batch_fn) if (batch_tp + batch_fn) > 0 else 0
            batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0

            # Update progress bar with current metrics
            batch_loss = val_loss.item()
            pbar.set_postfix({
                'Loss': f'{batch_loss:.3f}',
                'Prec': f'{batch_precision:.3f}',
                'Rec': f'{batch_recall:.3f}',
                'F1': f'{batch_f1:.3f}',
                'GT': batch_gt_objects,
                'Pred': batch_pred_objects
            })

    # Calculate overall metrics
    avg_loss = total_loss / len(data_loader)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Detection rate (how many images had at least one detection)
    detection_rate = total_pred_objects / num_samples if num_samples > 0 else 0

    print(f"    Validation Summary:")
    print(f"    - Total GT objects: {total_gt_objects}")
    print(f"    - Total Predictions (conf>0.5): {total_pred_objects}")
    print(f"    - True Positives: {total_tp}")
    print(f"    - False Positives: {total_fp}")
    print(f"    - False Negatives: {total_fn}")

    return avg_loss, precision, recall, f1_score, detection_rate


