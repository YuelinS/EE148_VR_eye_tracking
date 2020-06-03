# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:11:20 2020

@author: shiyl
"""
import numpy as np

x = np.load('D:/git/EE148/results_project/session0/orig_time_per_im.npy')

[train_batch_losses,val_losses] = np.load('D:/git/EE148/results_project/session0/orig_loss_train.npy',allow_pickle=True)
