import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os

class EyeTrackingDataset(Dataset):
    """Fove eye tracking dataset."""

    def __init__(self, path_pos, dir_images, transform=None):
        self.path_pos = path_pos
        self.dir_images = dir_images
        self.transform = transform
        self.dt_pos = np.dtype([('timestamp', np.int64), ('x', np.single), ('y', np.single), ('z', np.single)])
        pos_raw = np.fromfile(self.path_pos, dtype=self.dt_pos, offset=256)
        self.pos = np.hstack([pos_raw['x'].reshape(-1, 1), pos_raw['y'].reshape(-1, 1), pos_raw['z'].reshape(-1, 1)])#.astype(np.double)

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path_image = os.path.join(self.dir_images, f'{idx}.png')
        image = Image.open(path_image)
        if self.transform:
            image = self.transform(image)
        return (image, self.pos[idx, :])