import numpy as np
from skimage import io, color
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os
from PIL import Image

class EyeTrackingDataset(Dataset):
    """Fove eye tracking dataset."""

    def __init__(self, path_pos, dir_images, transform=None):
        self.path_pos = path_pos
        self.dir_images = dir_images
        self.transform = transform
        self.dt_pos = np.dtype([('timestamp', np.int64), ('x', np.single), ('y', np.single), ('z', np.single)])
        pos_raw = np.fromfile(self.path_pos, dtype=self.dt_pos, offset=256)
        self.pos = np.hstack([pos_raw['x'].reshape(-1, 1), pos_raw['y'].reshape(-1, 1), pos_raw['z'].reshape(-1, 1)]).astype(np.double)

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path_image = os.path.join(self.dir_images, f'{idx}.png')
        image = Image.fromarray(color.rgb2gray(io.imread(path_image)))
        if self.transform:
            image = self.transform(image)
        return (image, self.pos[idx, :])
    
#%%
# =============================================================================
# import matplotlib.pyplot as plt
# import numpy as np
# 
# path_pos = '../../data/Fixation Training Pos.bin'
# dir_images = '../../data/Fixation Training Images'   
#  
# img_transform = transforms.Compose([
#         # transforms.RandomRotation(10),
#         # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
#         transforms.Resize((24,64)),
#         transforms.ToTensor()])
# 
# face_dataset = EyeTrackingDataset(path_pos, dir_images, transform=img_transform)
# 
# 
# fig = plt.figure()
# 
# for i in range(3):
#     img,target = face_dataset[i]
#     im = img.numpy()*255
#     print(i, img.size,target)
# 
#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(im.transpose(1,2,0)[:,:,0],cmap = 'gray')   
#     plt.show()
# =============================================================================
