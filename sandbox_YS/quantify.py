# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:37:07 2020

@author: shiyl
"""
import os
import numpy as np
import matplotlib.pyplot as plt


dfd1 = 'D:/git/EE148/results_project/session0/'  
dfd2 = 'D:/git/EE148/results_project/session0/'  

rfd = 'D:/git/EE148/results_project/visualize/'    


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
    
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


#%% Inspect predictions

[trues,preds] = np.load(dfd1 + 'orig_model_prediction.npy',allow_pickle=True)

trues = [sample for batch in trues for sample in batch]
preds = [sample for batch in preds for sample in batch]
    
dist = [np.linalg.norm(trues[isample] - preds[isample]) for isample in range(len(trues))]
angle = [angle_between(trues[isample], preds[isample]) for isample in range(len(trues))]

np.mean(dist)
# 1.014827

np.mean(angle)
# 0.078

# np.save(rfd + 'orig_s0_model_prediction.png')


#%% num of params

import torch

params = torch.load('./orig_save_model.pt',map_location=torch.device('cpu'))

np.sum([params[name].numel() for name in params][:12])
# 68483