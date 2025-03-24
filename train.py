import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from dataset import FITSDataset
from skimage.metrics import structural_similarity as ssim_score
from skimage.metrics import peak_signal_noise_ratio as psnr_score

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    ssim_total, psnr_total = 0.0, 0.0
    for i in range(preds.shape[0]):
        pred_img = preds[i, 0]
        target_img = targets[i, 0]
        ssim_total += ssim_score(pred_img, target_img, data_range=1.0)
        psnr_total += psnr_score(target_img, pred_img, data_range=1.0)
    return ssim_total / preds.shape[0], psnr_total / preds.shape[0]


def save_predictions(images, outputs, filenames, epoch):
    outputs = torch.clamp(outputs, 0.0, 1.0)
    os.makedirs(f"{OUTPUT_DIR}/epoch{epoch}", exist_ok=True)
    for i in range(images.shape[0]):
        pred = outputs[i]
        save_path = os.path.join(f"{OUTPUT_DIR}/epoch{epoch}", f"{filenames[i]}.png")
        save_image(pred, save_path)
        print(f"Saved prediction: {save_path}")


def validate_unet(model, val_loader, loss_fn, epoch):
    model.eval()
    val_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.clamp(outputs, 0.0, 1.0)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            ssim_batch, psnr_batch = compute_metrics(outputs, masks)
            total_ssim += ssim_batch
            total_psnr += psnr_batch

            if batch_idx == 0:
                save_predictions(images, outputs, filenames, epoch)

    avg_val_loss = val_loss / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB")

    model.train()
    return avg_val_loss, avg_ssim, avg_psnr


def train_unet(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20):
    model.train()

    train_losses, val_losses = [], []
    ssim_scores, psnr_scores = [], []

    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation
        if (epoch + 1) % 5 == 0:
            val_loss, val_ssim, val_psnr = validate_unet(model, val_loader, loss_fn, epoch)
            val_losses.append(val_loss)
            ssim_scores.append(val_ssim)
            psnr_scores.append(val_psnr)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

            scheduler.step(val_loss)
        else:
            val_losses.append(val_losses[-1] if val_losses else avg_train_loss)
            ssim_scores.append(ssim_scores[-1] if ssim_scores else 0)
            psnr_scores.append(psnr_scores[-1] if psnr_scores else 0)

    print("Training Complete!")

    # --- Save Final Static Plot ---
    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Loss
    axs[0].plot(epochs, train_losses, label="Train Loss", color='blue')
    axs[0].plot(epochs, val_losses, label="Validation Loss", color='orange')
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # 2. SSIM
    axs[1].plot(epochs, ssim_scores, label="Validation SSIM", color='green')
    axs[1].set_title("Validation SSIM")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SSIM")
    axs[1].legend()

    # 3. PSNR
    axs[2].plot(epochs, psnr_scores, label="Validation PSNR", color='purple')
    axs[2].set_title("Validation PSNR")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("PSNR (dB)")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_plot.png"))
    plt.close()


def train():
    print("Device detected:", device)

    model = UNet(in_channels=1, out_channels=1).to(device)
    loss_fn = lambda output, target: 0.8 * nn.L1Loss()(output, target) + 0.2 * nn.MSELoss()(output, target)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = FITSDataset('patches/train/raw', 'patches/train/cal')
    test_dataset = FITSDataset('patches/test/raw', 'patches/test/cal')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    train_unet(model, train_loader, test_loader, optimizer, loss_fn, num_epochs=50)
