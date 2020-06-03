# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:37:07 2020

@author: shiyl
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font',size = 25)
mpl.rc('lines',lw = 5)



dfd1 = 'D:/git/EE148/results_project/session0/'  
dfd2 = 'D:/git/EE148/results_project/session4/'  

rfd = 'D:/git/EE148/results_project/visualize/'    


#%% Training loss across epochs
from matplotlib.ticker import MaxNLocator

for i in range(3):
    
    dpos = [dfd1+'orig_', dfd2+'tran_', dfd2+'orig4_'][i]
    rpos = [rfd + 'orig_s0_', rfd+'tran_s4_', rfd + 'orig4_s4_'][i]
    
    [train_batch_losses,val_losses] = np.load(dpos + 'loss_train.npy',allow_pickle=True)
    # test_loss = np.load(dpos + 'loss_test.npy')
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))  
    ax.plot(train_batch_losses,marker=".")
    ax.plot(val_losses,marker=".")
    ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('# Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['Train loss','Validation loss'])
    # ax.set_title(f'Test Loss: {test_loss:.3f}')
    
    plt.savefig(rpos + 'training_loss_curve.png')


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

params = torch.load('./orig_save_model.pt',map_location=torch.device('cpu'))
c1w = params['conv1.weight'].numpy()


fig,axs = plt.subplots(3,3,figsize=(15,15))
axs = axs.ravel()
for i in range(9):  
    axs[i].imshow(np.squeeze(c1w[i,0]),cmap = 'gray')
    
plt.savefig(rfd + 'orig_s0_kernels.png')



#%% Inspect predictions
# from mpl_toolkits.mplot3d import Axes3D

[trues,preds] = np.load(dfd1 + 'orig_model_prediction.npy',allow_pickle=True)
batch_size = len(trues[0])

ibatch = 0
pairs = np.dstack((trues[ibatch], preds[ibatch]))


fig = plt.figure(figsize=(15, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
mpl.rc('lines',lw = 2)

for k in range(batch_size):
    ax.plot(pairs[k, 0, :], pairs[k, 2, :], pairs[k, 1, :])
    
ax.invert_yaxis()
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.set_xlim(-5, 5)
ax.view_init(30, 30)

plt.savefig(rfd + 'orig_s0_model_prediction.png')



#%%% Transfer learning - s4

# traning history 1st epoch - orig s0 vs tran s4
 
    
 
    
 
    
 
    


