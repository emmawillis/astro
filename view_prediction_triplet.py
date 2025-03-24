import os
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

def load_fits_image(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data  # Do not normalize — preserve scientific structure

def display_triplet(pred_path, raw_path, cal_path, out_path, title):
    # Load data
    pred = load_fits_image(pred_path)
    raw = load_fits_image(raw_path)
    cal = load_fits_image(cal_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw, cmap='gray', origin='lower')
    axes[0].set_title("Raw Input")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap='gray', origin='lower')
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(cal, cmap='gray', origin='lower')
    axes[2].set_title("Ground Truth (Calibrated)")
    axes[2].axis("off")

    plt.suptitle(f"{title}\n{os.path.basename(pred_path)}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {out_path}")

def generate_all_triplets(pred_dir):
    epoch = os.path.basename(pred_dir.rstrip("/"))
    out_dir = os.path.join("visualizations", epoch)
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith((".fit", ".fits")):
            continue

        pred_path = os.path.join(pred_dir, fname)
        raw_path = os.path.join("patches", "test", "raw", fname)
        cal_path = os.path.join("patches", "test", "cal", fname)
        out_path = os.path.join(out_dir, f"{fname.replace('.fits', '').replace('.fit', '')}_triplet.png")

        if not (os.path.exists(raw_path) and os.path.exists(cal_path)):
            print(f"⚠️ Skipping missing file: {fname}")
            continue

        display_triplet(pred_path, raw_path, cal_path, out_path, title=f"{epoch} result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate triplet plots for all predictions in a directory")
    parser.add_argument("pred_dir", type=str, help="Directory containing predicted FITS files")
    args = parser.parse_args()

    generate_all_triplets(args.pred_dir)
