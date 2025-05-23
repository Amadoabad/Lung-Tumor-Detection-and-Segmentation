import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, loss_fn, device):

    model.train()
    train_losses = 0
    loop = tqdm(data_loader)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)
        train_losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

    loss = train_losses/len(data_loader)
    return loss

def validate_model(loader, model, loss_fn, device="cuda", alpha=0.7, beta=0.3):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    tversky_score = 0
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Accuracy
            num_correct += (preds == masks).sum().item()
            num_pixels += torch.numel(preds)

            # Dice
            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            dice_score += (2 * intersection + 1e-8) / (union + 1e-8)

            # Tversky
            TP = (preds * masks).sum().item()
            FP = (preds * (1 - masks)).sum().item()
            FN = ((1 - preds) * masks).sum().item()
            tversky = TP / (TP + alpha * FP + beta * FN + 1e-8)
            tversky_score += tversky

    avg_loss = total_loss / len(loader)
    avg_dice = dice_score / len(loader)
    avg_tversky = tversky_score / len(loader)
    acc = num_correct / num_pixels * 100

    print(f"[Validation] Acc: {acc:.2f}%, Dice: {avg_dice:.4f}, Tversky: {avg_tversky:.4f}, Loss: {avg_loss:.4f}")
    return avg_loss, avg_dice, avg_tversky

def train(model, data_loader,val_loader, optimizer, schedualer, loss_fn, device, epochs, save_path="best_model.pth"):
    best_tversky = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train
        epoch_loss = train_one_epoch(model, data_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Validation
        val_loss, val_dice, val_tversky = validate_model(val_loader, model, loss_fn, device)
        schedualer.step(val_loss)
        
        # Save the best model
        if val_tversky > best_tversky:
            best_tversky = val_tversky
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with Tversky: {best_tversky:.4f}")