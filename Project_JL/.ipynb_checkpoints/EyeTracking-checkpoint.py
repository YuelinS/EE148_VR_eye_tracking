import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os

class EyeTrackingDataset(Dataset):
    """Fove eye tracking dataset."""

    def __init__(self, path_pos, dir_images, transform=None, polar=False):
        self.path_pos = path_pos
        self.dir_images = dir_images
        self.transform = transform
        self.polar = polar
        self.dt_pos = np.dtype([('timestamp', np.int64), ('x', np.single), ('y', np.single), ('z', np.single), 
                               ('x_fove', np.single), ('y_fove', np.single), ('z_fove', np.single), ('is_open', np.bool)])
        pos_raw = np.fromfile(self.path_pos, dtype=self.dt_pos, offset=256)
        self.pos = np.hstack([pos_raw['x'].reshape(-1, 1), pos_raw['y'].reshape(-1, 1), pos_raw['z'].reshape(-1, 1), pos_raw['is_open'].reshape(-1, 1)])
        self.pos_fove = np.hstack([pos_raw['x_fove'].reshape(-1, 1), pos_raw['y_fove'].reshape(-1, 1), pos_raw['z_fove'].reshape(-1, 1), pos_raw['is_open'].reshape(-1, 1)])
        self.pos_polar = self.cart2polar(self.pos)
        self.pos_fove_polar = self.cart2polar(self.pos_fove)

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
        if self.polar:
            pos = self.pos_polar[idx, :]
            pos_fove = self.pos_fove_polar[idx, :]
        else:
            pos = self.pos[idx, :]
            pos_fove = self.pos_fove[idx, :]
        return (image, pos, pos_fove)
    
    def cart2polar(self, cart):
        '''
        Covert left-hand cartesian coords to polar coords
        cart is in [x, y, z]
        polar is in [azimuth, elevation, distance]
        '''
        polar = cart.copy()
        polar[:, 0] = np.rad2deg(np.arctan2(cart[:, 0], cart[:, 2]))
        polar[:, 1] = np.rad2deg(np.arctan2(cart[:, 1], cart[:, 2]))
        polar[:, 2] = (cart[:, 0]**2+cart[:, 1]**2+cart[:, 2]**2)**0.5
        return polar