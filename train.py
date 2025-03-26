import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from dataset import FITSDataset
from skimage.metrics import structural_similarity as ssim_score
from skimage.metrics import peak_signal_noise_ratio as psnr_score
from astropy.io import fits


# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "predictions"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_predictions(outputs, filenames, epoch):
    os.makedirs(f"{OUTPUT_DIR}/epoch{epoch}", exist_ok=True)
    for i in range(outputs.shape[0]):
        pred = outputs[i].squeeze().cpu().numpy()
        base_name = filenames[i]
        if not base_name.endswith((".fits", ".fit")):
            base_name += ".fits"
        save_path = os.path.join(f"{OUTPUT_DIR}/epoch{epoch}", base_name)
        fits.writeto(save_path, pred, overwrite=True)
        print(f"Saved prediction: {save_path}")

def validate_unet(model, val_loader, loss_fn, epoch):
    model.eval()
    val_loss, total_ssim, total_psnr = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            # Optional metrics (for monitoring)
            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                pred_img = outputs_np[i, 0]
                target_img = masks_np[i, 0]
                psnr = psnr_score(target_img, pred_img, data_range=target_img.max() - target_img.min())
                total_psnr += psnr

            if batch_idx == 0:
                save_predictions(outputs, filenames, epoch)

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

    model.train()
    return avg_val_loss, avg_psnr

def train_unet(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=100):
    train_losses, val_losses, psnr_scores = [], [], []
    best_loss = float("inf")
    patience, patience_counter = 10, 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            val_loss, psnr = validate_unet(model, val_loader, loss_fn, epoch)
            val_losses.append(val_loss)
            psnr_scores.append(psnr)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                print(f"✅ Saved best model (epoch {epoch+1})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("⏹️ Early stopping triggered, but continuing anyway")
                # break

            scheduler.step(val_loss)

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(range(4, 4+5*len(val_losses), 5), val_losses, label="Val Loss")
    plt.plot(range(4, 4+5*len(psnr_scores), 5), psnr_scores, label="PSNR (dB)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
    plt.close()

def train():
    print(f"Using device: {device}")
    model = UNet(in_channels=1, out_channels=1).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = FITSDataset("patches/train/raw", "patches/train/cal", augment=True)
    val_dataset = FITSDataset("patches/test/raw", "patches/test/cal", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    train_unet(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=200)
