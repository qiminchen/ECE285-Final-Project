#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:55:37 2019

@author: qiminchen
"""

import matplotlib.pyplot as plt
import numpy as np

def convert(img):
    
    img = img.to('cpu').detach().numpy()
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    img[img < 0] = 0
    img[img > 1] = 1
    
    return img

def display(img1, img2):
    
    img1 = convert(img1)
    img2 = convert(img2)
    
    fig, ax = plt.subplots(1, 2, figsize=(9,6))
    ax[0].set_title('Style')
    ax[0].imshow(img1)
    ax[1].set_title('Content')
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')
    
def display_style_content(imgs, types):
    
    fig, ax = plt.subplots(1, len(types), figsize=(len(types)*4,len(types)*4))
    for i, style in enumerate(types):
        
        img = convert(imgs[i][0])
        ax[i].set_title(style)
        ax[i].imshow(img)
        ax[i].axis('off')
    
    
    
    
    