import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os

class EyeTrackingDataset(Dataset):
    """Fove eye tracking dataset."""

    def __init__(self, path_pos, dir_images, transform=None, polar=False, eye='both', coord_3d=True, combine_open=False, use_fove_gaze=False):
        '''
        polar: whether if to convert 3d coords into polar
        eye: 'both', 'left' or 'right'
        coord_3d: wheter if to return 3d gaze vector
        combine_open: combine the opening into the vector, making the coordinates 4d (or 3d if coord_3d is False)
            otherwise return separately
        use_fove_gaze: for a single eye, whether if to use Fove's 
        '''
        if eye=='both' and not coord_3d:
            raise ValueError('2d coordinates is not available for both-eye configuration.')
        self.w = 640
        self.h = 240
        self.path_pos = path_pos
        self.dir_images = dir_images
        self.transform = transform
        self.polar = polar
        self.eye = eye
        self.coord_3d = coord_3d
        self.combine_open = combine_open
        self.use_fove_gaze = use_fove_gaze
        self.dt_pos = np.dtype([
            ('timestamp', np.int64), 
            ('pos_fix', np.single, 3), 
            ('left_offset', np.single, 3),
            ('right_offset', np.single, 3),
            ('left_gaze', np.single, 3),
            ('right_gaze', np.single, 3),
            ('left_gaze_2d', np.single, 2),
            ('right_gaze_2d', np.single, 2),
            ('is_left_open', np.bool), 
            ('is_right_open', np.bool), 
        ])
        pos_raw = np.fromfile(path_pos, dtype=self.dt_pos, offset=256)
        self.pos_fix = pos_raw['pos_fix']
        self.pos_fix_polar = self.cart2polar(self.pos_fix)
        self.left_offset = pos_raw['left_offset'][0, :]
        self.right_offset = pos_raw['left_offset'][0, :]
        self.left_gaze = pos_raw['left_gaze'] if self.use_fove_gaze else self.pos_fix-self.left_offset
        self.left_gaze_polar = self.cart2polar(self.left_gaze)
        self.right_gaze = pos_raw['right_gaze'] if self.use_fove_gaze else self.pos_fix-self.right_offset
        self.right_gaze_polar = self.cart2polar(self.right_gaze)
        self.left_gaze_2d = pos_raw['left_gaze_2d']
        self.right_gaze_2d = pos_raw['right_gaze_2d']
        self.is_left_open = pos_raw['is_left_open']
        self.is_right_open = pos_raw['is_right_open']
        if polar:
            self._pos = self.pos_fix_polar
            self._left_gaze = self.left_gaze_polar
            self._right_gaze = self.right_gaze_polar
        else:
            self._pos = self.pos_fix
            self._left_gaze = self.left_gaze
            self._right_gaze = self.right_gaze
        if not coord_3d:
            self._left_gaze = self.left_gaze_2d
            self._right_gaze = self.right_gaze_2d
        if eye=='both':
            self.crop = (0, 0, self.w, self.h)
            self._pos = self._pos
            self._is_open = np.logical_and(self.is_left_open, self.is_right_open)
        elif eye=='left':
            self.crop = (0, 0, self.w//2, self.h)
            self._pos = self._left_gaze
            self._is_open = self.is_left_open
        elif eye=='right':
            self.crop = (self.w//2, 0, self.w, self.h)
            self._pos = self._right_gaze
            self._is_open = self.is_right_open
        else:
            raise ValueError('Eye must be either both or left or right.')
        if combine_open:
            self._pos = np.hstack((self._pos, self._is_open.reshape(-1, 1)))
    
    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path_image = os.path.join(self.dir_images, f'{idx}.png')
        image = Image.open(path_image)
        image = image.crop(self.crop)
        if self.transform:
            image = self.transform(image)
        pos = self._pos[idx, :]
        is_open = self._is_open[idx]
        if self.combine_open:
            return (image, pos)
        else:
            return (image, pos, is_open)
    
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
    
    def polar2cart(self, polar):
        '''
        Covert polar coords to left-hand cartesian coords
        polar is in [azimuth, elevation, distance]
        cart is in [x, y, z]
        '''
        cart = polar.copy()
        polar_tan = np.tan(np.deg2rad(polar))
        back = (np.logical_or(polar[:, 0]>-90, polar[:, 0]<90)-0.5)*2
        cart[:, 2] = polar[:, 2]/np.sqrt(1+polar_tan[:, 0]**2+polar_tan[:, 1]**2)*back
        cart[:, 0] = cart[:, 2]*polar_tan[:, 0]
        cart[:, 1] = cart[:, 2]*polar_tan[:, 1]
        return cart
    
    def normalize(self, cart):
        return cart/((cart**2).sum(1)**0.5)