from tqdm import tqdm
import torch
import sys
import os
from models.Mask_Rcnn import get_model
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import evaluate
from config import device, train_loader, val_loader, num_epochs, learning_rate

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



best_f1 = 0
best_precision = 0
best_recall = 0

# Create model
model = get_model(num_classes=2)  # Background + Tumor
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

for epoch in range(num_epochs):
    # Train
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

    # Evaluate
    val_loss, val_precision, val_recall, val_f1, detection_rate = evaluate(model, val_loader, device, epoch)

    # Update learning rate
    scheduler.step()

    # Print epoch summary
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}')
    print(f'  Val Loss: {val_loss:.4f}')
    print(f'  Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
    print(f'  Detection Rate: {detection_rate:.2f} pred/image')
    print(f'  Best F1: {best_f1:.4f} (P: {best_precision:.4f}, R: {best_recall:.4f})')

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_precision = val_precision
        best_recall = val_recall
        torch.save(model.state_dict(), 'best_tumor_segmentation_model.pth')
        print(f'  *** New best model saved! ***')

    print('-' * 80)

print(f'\nTraining completed!')
print(f'Best F1: {best_f1:.4f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f})')


