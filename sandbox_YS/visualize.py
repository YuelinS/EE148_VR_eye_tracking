# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:37:07 2020

@author: shiyl
"""
import os
import numpy as np
import matplotlib.pyplot as plt

rfd = 'D:/git/EE148/results_project/'    


#%% Training loss across epochs
from matplotlib.ticker import MaxNLocator

[train_batch_losses,val_losses] = np.load(rfd+'loss_train.npy',allow_pickle=True)
test_loss = np.load(rfd+'loss_test.npy')

fig, ax = plt.subplots(1, 1, figsize=(20, 10))  
ax.plot(train_batch_losses,marker=".")
ax.plot(val_losses,marker=".")
ax.grid()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('# Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Test Loss: {test_loss:.3f}')

plt.savefig(rfd + 'training_loss_curve.png')


#%% Learning curve

file_names = sorted(os.listdir(rfd)) 

train_accs, test_accs = [], []
parts = [1,2,4,8,16][::-1] 

for partition in parts:
    
    train_fn = [f for f in file_names if 'part'+str(partition) in f and 'train' in f] 
    test_fn = [f for f in file_names if 'part'+str(partition) in f and 'test' in f] 
    
    train_acc = np.load(rfd + train_fn[0])[2,-1]
    test_acc =  np.load(rfd + test_fn[0])
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)


n_train = 60000*.15
n_parts = [n_train/part for part in parts]


plt.figure(figsize=(20, 10))  
plt.loglog(n_parts,train_accs,marker=".")
plt.loglog(n_parts,test_accs,marker=".")
plt.grid()
# plt.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(label=[str(part) for part in parts])
plt.xlabel('Training data size')
plt.ylabel('Acccuracy')
plt.legend({'Train','Test'})
# plt.set_title('Loss')


plt.savefig(rfd + 'data_partition.png')



#%% visualize kernels

import torch

params = torch.load('./eye_tracking_model2.pt',map_location=torch.device('cpu'))
c1w = params['conv1.weight'].numpy()


fig,axs = plt.subplots(3,3,figsize=(15,15))
axs = axs.ravel()
for i in range(8):  
    axs[i].imshow(np.squeeze(c1w[i,0]),cmap = 'gray')
    # axs[i].set_title(f'True: {trues[i]}, Pred: {preds[i][0]}')
    
plt.savefig(rfd + 'kernels.png')



#%% Inspect predictions

from mpl_toolkits.mplot3d import Axes3D

files = np.load(rfd + 'model_prediction.npz')

trues = files['arr_0']
preds = np.squeeze(files['arr_1'])
batch_size = np.shape(trues)[1]

ibatch = 0
pairs = np.dstack((trues[ibatch,:,:], preds[ibatch,:,:]))


fig = plt.figure(figsize=(15, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')

for k in range(batch_size):
    ax.plot(pairs[k, 0, :], pairs[k, 1, :], pairs[k, 2, :])
    
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(45, 60)

plt.savefig(rfd + 'model_prediction.png')



#%% Incorrect examples

[imgs,trues,preds] = np.load(rfd + 'incorrect_examples.npy')

fig,axs = plt.subplots(3,3,figsize=(15,15))
axs = axs.ravel()
for i in range(9):  
    axs[i].imshow(np.squeeze(imgs[i]),cmap = 'gray')
    axs[i].set_title(f'True: {trues[i]}, Pred: {preds[i][0]}')
    
plt.savefig(rfd + 'incorrect_examples.png')



#%% feature
from sklearn.manifold import TSNE

files = np.load(rfd + 'feature_data.npz')

features = files['arr_0']
labels = files['arr_1'].ravel()

## t-SNE
feat = features.reshape(-1,64)
X_embedded = TSNE(n_components=2).fit_transform(feat)

color = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25],[0, 130, 200], [245, 130, 48], [230, 190, 255],
                  [170, 110, 40], [255, 250, 200], [210, 245, 60], [250, 190, 190]])

fig, ax = plt.subplots()
for digit in range(10):

    x = X_embedded[labels==digit,0]
    y = X_embedded[labels==digit,1] 
    ax.scatter(x, y, alpha=0.8, c=tuple(color[digit]/255), edgecolors='none', s=5,label=digit) #,cmap = 'winter') #, 

plt.legend()
plt.savefig(rfd + 't-SNE.png')


#%% 8neighbors
from scipy.spatial.distance import euclidean as euclidian
import pickle

ims_pos = []

for k in range(5):
    
    x0 = feat[k]
    ds = [euclidian(x0, xi) for xi in feat]
    d9 = sorted(ds)[:9]
    p9 = [ds.index(d) for d in d9]
    
    im_pos = [[p//1000,p%1000] for p in p9]
    ims_pos.append(im_pos)

# np.save('8neighbors_pos',ims_pos)
with open(rfd+"8neighbors_pos.txt", "wb") as fp:   #Pickling
   pickle.dump(ims_pos, fp)




## images
imgs = np.load(rfd+'8neighbors_img.npy')

fig,axs = plt.subplots(5,9,figsize=(25,15))

for i in range(5): 
    for j in range(9):
        
        axs[i,j].imshow(np.squeeze(imgs[i,j]),cmap = 'gray')
    
plt.savefig(rfd + '8neighbors_grid.png')


