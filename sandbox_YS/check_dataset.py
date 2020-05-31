import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load 

# dir_images = '../../data/images0'
path_pos = '../../data/pos0.bin'
dt_pos = np.dtype([('timestamp', np.int64), ('x', np.single), ('y', np.single), ('z', np.single), 
                       ('x_fove', np.single), ('y_fove', np.single), ('z_fove', np.single), ('is_open', np.bool)])
pos_raw = np.fromfile(path_pos, dtype=dt_pos, offset=256)
pos = np.hstack([pos_raw['x'].reshape(-1, 1), pos_raw['y'].reshape(-1, 1), pos_raw['z'].reshape(-1, 1), pos_raw['is_open'].reshape(-1, 1)])
pos_fove = np.hstack([pos_raw['x_fove'].reshape(-1, 1), pos_raw['y_fove'].reshape(-1, 1), pos_raw['z_fove'].reshape(-1, 1), pos_raw['is_open'].reshape(-1, 1)])

len_data = len(pos)



#%%
     
# check scan pattern

fig = plt.figure(figsize=(15, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')

for k in range(0,len_data,100):
    target = pos[k][:3]
    # print(target)
    ax.scatter3D(target[0], target[1], target[2])
    
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.view_init(-90, 45)
# plt.savefig(rfd + 'model_prediction.png')


# compare parameters -- single line of sight

fig = plt.figure(figsize=(15, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')

for k in range(0,len_data,100):
    target = pos[k][:3]
    left_sight = target - [-3,0,0]
    # print(target)
    ax.scatter3D(left_sight[0], left_sight[1], left_sight[2])
    
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

















