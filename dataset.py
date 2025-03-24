import torch
from torch.utils.data import Dataset
import os
import numpy as np
from astropy.io import fits

class FITSDataset(Dataset):
    def __init__(self, raw_dir, cal_dir, raw_transform=None, cal_transform=None):
        self.raw_filenames = sorted([
            f for f in os.listdir(raw_dir) 
            if f.endswith(('.fit', '.fits')) and os.path.exists(os.path.join(cal_dir, f))
        ])
        self.raw_dir = raw_dir
        self.cal_dir = cal_dir
        self.raw_transform = raw_transform
        self.cal_transform = cal_transform
    
    def __len__(self):
        return len(self.raw_filenames)

    def _load_fits(self, filepath):
        """Loads a FITS file and returns a (1, 256, 256) tensor normalized to [0, 1]."""
        with fits.open(filepath) as hdul:
            data = hdul[0].data  # Extract the image data

        data = np.array(data, dtype=np.float32)

        # Normalize to [0,1]
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = data * 0  # Uniform fallback

        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {data.shape} for file {filepath}")

        # Add channel dimension → shape: (1, 256, 256)
        data = data[np.newaxis, :, :]

        return data

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_filenames[idx])
        cal_path = os.path.join(self.cal_dir, self.raw_filenames[idx])

        # Safety check
        if not os.path.exists(raw_path) or not os.path.exists(cal_path):
            if not os.path.exists(raw_path):
                print(f"!!! RAW File not found: {raw_path}")
            if not os.path.exists(cal_path):
                print(f"!!! CAL File not found: {cal_path}")

        raw_image = self._load_fits(raw_path)
        cal_image = self._load_fits(cal_path)

        # Convert to tensors
        raw_image = torch.tensor(raw_image, dtype=torch.float32)    # Shape: (1, 256, 256)
        cal_image = torch.tensor(cal_image, dtype=torch.float32)    # Shape: (1, 256, 256)

        # Optional transforms
        if self.raw_transform:
            raw_image = self.raw_transform(raw_image)
        if self.cal_transform:
            cal_image = self.cal_transform(cal_image)

        return raw_image, cal_image, self.raw_filenames[idx]
