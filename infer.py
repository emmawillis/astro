import os
import torch
import numpy as np
from astropy.io import fits
from unet import UNet
from math import ceil

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pt"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATCH_SIZE = 256


def load_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {data.shape}")
    return data


def save_fits(path, data):
    fits.writeto(path, data.astype(np.float32), overwrite=True)


def preprocess(patch):
    """Convert patch to tensor with shape (1, 1, H, W)"""
    return torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def postprocess(output_tensor):
    """Convert output tensor (1, 1, H, W) -> (H, W) NumPy array"""
    return output_tensor.squeeze().detach().cpu().numpy()


def pad_image(image, patch_size):
    H, W = image.shape
    pad_h = ceil(H / patch_size) * patch_size - H
    pad_w = ceil(W / patch_size) * patch_size - W
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return padded, (H, W)


def unpad_image(image, original_shape):
    H, W = original_shape
    return image[:H, :W]


def run_inference(fits_path):
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and pad image
    input_img = load_fits(fits_path)
    padded_img, original_shape = pad_image(input_img, PATCH_SIZE)
    H_pad, W_pad = padded_img.shape

    output_img = np.zeros_like(padded_img)

    # Slide over patches
    for y in range(0, H_pad, PATCH_SIZE):
        for x in range(0, W_pad, PATCH_SIZE):
            patch = padded_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            input_tensor = preprocess(patch)

            with torch.no_grad():
                output_tensor = model(input_tensor)

            denoised_patch = postprocess(output_tensor)
            output_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = denoised_patch

    # Crop back to original size
    final_img = unpad_image(output_img, original_shape)

    # Save result
    base_name = os.path.basename(fits_path).replace(".fits", "").replace(".fit", "")
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_AUTOCORRECTED.fits")
    save_fits(output_path, final_img)

    print(f"✅ Saved cleaned image to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a raw FITS file")
    parser.add_argument("fits_path", type=str, help="Path to the raw input FITS image")
    args = parser.parse_args()

    path = args.fits_path.replace("outputs/", "patches/test/raw/")
    print("!! ", path)
    run_inference(args.fits_path)
