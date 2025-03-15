import torch
from torch.utils.data import Dataset
import os
import numpy as np
from astropy.io import fits

class FITSDataset(Dataset):
    def __init__(self, raw_dir, cal_dir, raw_transform=None, cal_transform=None):
        self.raw_filenames = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.fit', '.fits'))])
        self.raw_dir = raw_dir
        self.cal_dir = cal_dir
        self.raw_transform = raw_transform
        self.cal_transform = cal_transform
    
    def __len__(self):
        return len(self.raw_filenames)

    def _load_fits(self, filepath):
        """Loads a FITS file and ensures it has shape (256, 256, 3)."""
        with fits.open(filepath) as hdul:
            data = hdul[0].data  # Extract the image data

        data = np.array(data, dtype=np.float32)

        # Normalize data to [0,1] range (recommended)
        data_min, data_max = data.min(), data.max()   # TODO do we need to normalize?
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = data * 0  # handle edge case: uniform image

        # Since all are single-channel (256, 256), duplicate to 3 channels # TODO - do we need 3 channels here??
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)  # Shape: (256,256,3)
        else:
            raise ValueError(f"Unexpected shape {data.shape} for file {filepath}")

        return data

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_filenames[idx])
        cal_path = os.path.join(self.cal_dir, self.raw_filenames[idx])

        # Load raw and calibrated images
        raw_image = self._load_fits(raw_path)
        cal_image = self._load_fits(cal_path)

        # Convert to PyTorch tensor and reorder to (channels, height, width)
        raw_image = torch.tensor(raw_image, dtype=torch.float32).permute(2, 0, 1)  # (3,256,256)
        cal_image = torch.tensor(cal_image, dtype=torch.float32).permute(2, 0, 1)  # (3,256,256)

        # Apply optional transforms
        if self.raw_transform:
            raw_image = self.raw_transform(raw_image)
        if self.cal_transform:
            cal_image = self.cal_transform(cal_image)

        return raw_image, cal_image, self.raw_filenames[idx]
