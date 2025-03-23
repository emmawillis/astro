import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress bar
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
from unet import UNet

from dataset import FITSDataset

# Device configuration (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OUTPUT_DIR = "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_predictions(images, outputs, filenames, epoch):
    """
    Save predictions as images.
    Args:
        images: Input images (batch, C, H, W)
        outputs: Predicted masks (batch, H, W)
        filenames: List of original filenames to match ground truth
        epoch: Current epoch (for tracking)
    """
    outputs = torch.argmax(outputs, dim=1)  # Convert from (batch, 4, H, W) -> (batch, H, W)

    print("Unique values in predicted masks:", torch.unique(outputs))  # Check class distribution

    for i in range(images.shape[0]):
        pred_mask = outputs[i].cpu().float()  # Convert to float for saving
        os.makedirs(f"{OUTPUT_DIR}/epoch{epoch}", exist_ok=True)
        save_path = os.path.join(f"{OUTPUT_DIR}/epoch{epoch}", f"{filenames[i]}.png")
        save_image(pred_mask.unsqueeze(0), save_path)  # Add channel dim for saving
        print(f"Saved: {save_path}")

def validate_unet(model, val_loader, loss_fn, epoch):
    model.eval()  # Set to evaluation mode
    val_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (images, masks, filenames) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            if batch_idx == 0:
                save_predictions(images, outputs, filenames, epoch)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    model.train()  # Switch back to training mode
    return val_loss / len(val_loader)

def train_unet(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20):
    model.train()  # Set to training mode

    val_loss = 0
    best_loss = float("inf")
    patience = 10  # Stop training if no improvement for 10 epochs
    patience_counter = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks, _filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(images)  # Output shape: (batch, 3, 256, 256)

            loss = loss_fn(outputs, masks)  # `masks` should have shape (batch, 256, 256)

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        # Run validation every few epochs
        if (epoch + 1) % 5 == 0:
            val_loss = validate_unet(model, val_loader, loss_fn, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

            # Reduce learning rate if validation loss stops improving
            scheduler.step(val_loss)

    print("Training Complete!")


def train():
    print("device IN submitit: ", device)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=3).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # mri_transform = transforms.Compose([
    #     transforms.Normalize(mean=[DATASET_PIXEL_MEAN], std=[DATASET_PIXEL_STD])
    # ]) # TODO normalize???

    train_dataset = FITSDataset('patches/train/raw', 'patches/train/cal')
    test_dataset = FITSDataset('patches/test/raw', 'patches/test/cal')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    train_unet(model, train_loader, test_loader, optimizer, loss_fn, num_epochs=100)
