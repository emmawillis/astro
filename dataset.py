import torch
from torch.utils.data import Dataset
import os
import numpy as np
from astropy.io import fits
import random

class FITSDataset(Dataset):
    def __init__(self, raw_dir, cal_dir, raw_transform=None, cal_transform=None, augment=False):
        self.raw_filenames = sorted([
            f for f in os.listdir(raw_dir)
            if f.endswith(('.fit', '.fits')) and os.path.exists(os.path.join(cal_dir, f))
        ])
        self.raw_dir = raw_dir
        self.cal_dir = cal_dir
        self.raw_transform = raw_transform
        self.cal_transform = cal_transform
        self.augment = augment

    def __len__(self):
        return len(self.raw_filenames)

    def _load_fits(self, filepath):
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {data.shape} for file {filepath}")

        # Add channel dimension (1, H, W)
        return data[np.newaxis, :, :]

    def _random_augment(self, raw_img, cal_img):
        if random.random() < 0.5:
            raw_img = np.flip(raw_img, axis=2)  # horizontal flip
            cal_img = np.flip(cal_img, axis=2)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            raw_img = np.rot90(raw_img, k=k, axes=(1, 2)).copy()
            cal_img = np.rot90(cal_img, k=k, axes=(1, 2)).copy()
        return raw_img, cal_img

    def __getitem__(self, idx):
        filename = self.raw_filenames[idx]
        raw_path = os.path.join(self.raw_dir, filename)
        cal_path = os.path.join(self.cal_dir, filename)

        raw_img = self._load_fits(raw_path)
        cal_img = self._load_fits(cal_path)

        if self.augment:
            raw_img, cal_img = self._random_augment(raw_img, cal_img)

        raw_tensor = torch.tensor(raw_img.copy(), dtype=torch.float32)
        cal_tensor = torch.tensor(cal_img.copy(), dtype=torch.float32)

        if self.raw_transform:
            raw_tensor = self.raw_transform(raw_tensor)
        if self.cal_transform:
            cal_tensor = self.cal_transform(cal_tensor)

        return raw_tensor, cal_tensor, filename
