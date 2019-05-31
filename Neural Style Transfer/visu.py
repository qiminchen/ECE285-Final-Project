#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:55:37 2019

@author: qiminchen
"""

import matplotlib.pyplot as plt
import numpy as np

def display(img1, img2):
    
    img1 = img1.to('cpu').detach().numpy()
    img1 = np.moveaxis(img1, [0, 1, 2], [2, 0, 1])
    img1[img1 < 0] = 0
    img1[img1 > 1] = 1
    img2 = img2.to('cpu').detach().numpy()
    img2 = np.moveaxis(img2, [0, 1, 2], [2, 0, 1])
    img2[img2 < 0] = 0
    img2[img2 > 1] = 1
    
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].set_title('Style')
    ax[0].imshow(img1)
    ax[1].set_title('Content')
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')