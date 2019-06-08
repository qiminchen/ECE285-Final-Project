#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:35:35 2019

@author: qiminchen
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

def get_transfered_imgs(path):
    
    images = []
    styles = ['paul','sam','symbolism_nicholas','willem']
    result_img = ['field1_fake_B.png', 'field2_fake_B.png', 'beach_fake_B.png']
    
    for s in styles:
        imgs_root_path = os.path.join(path,
                                      s ,'test_latest/images')
        for i in result_img:
            img = Image.open(os.path.join(
                imgs_root_path,i)).convert('RGB')
            images.append(img)
    
    original = []
    ori_img = ['field1.png', 'field2.jpg', 'beach.jpg']
    for o in ori_img:
        
        img = Image.open(os.path.join('data/testA',o)).convert('RGB')
        original.append(img)
    
    return original, images


def get_content_style():
    
    images = []
    ori_img = ['field1.png', 'field2.jpg', 'beach.jpg']
    sty_img = ['paul-style.png','sam-style.png','nicholas-style.png','willem-style.png']
    for o in ori_img:
        img = Image.open(os.path.join('data/testA',o)).convert('RGB')
        images.append(img)
    for s in sty_img:
        img = Image.open(os.path.join('data/styles',s)).convert('RGB')
        images.append(img)
        
    return images

def display(path):
    
    ori_img, tras_img = get_transfered_imgs(path)
    
    types = ['Input' , 'Paul', 'Sam', 'Nicholas', 'Willem']
    fig, ax = plt.subplots(3, 5, figsize=(20,12), constrained_layout=False)
    
    for i in range(3):
        ax[i][0].imshow(ori_img[i])
        ax[i][1].imshow(tras_img[i])
        ax[i][2].imshow(tras_img[i+3])
        ax[i][3].imshow(tras_img[i+6])
        ax[i][4].imshow(tras_img[i+9])
        
        for j in range(5):
            ax[i][j].axis('off')
           
        if i == 0:
            for j in range(5):
                ax[i][j].set_title(types[j],fontsize=16)
        
    plt.subplots_adjust(wspace=.001, hspace=.05)
    #plt.savefig('result_for_cyclegan.png')
    

def display_style_content():
    
    imgs = get_content_style()
    fig, ax = plt.subplots(1, len(imgs), figsize=(35,7))
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
        ax[i].axis('off')
    
    ax[0].set_title('Field',fontsize=22)
    ax[1].set_title('Prairie',fontsize=22)
    ax[2].set_title('Beach',fontsize=22)
    ax[3].set_title('Paul',fontsize=22)
    ax[4].set_title('Sam',fontsize=22)
    ax[5].set_title('Nicholas',fontsize=22)
    ax[6].set_title('Willem',fontsize=22)
    
    plt.subplots_adjust(wspace=.08, hspace=.05)
    #plt.savefig('input_for_cyclegan.png')
    
def display_recover(path):
    
    images = []
    imgs = ['nicholas-style.png','nicholas-field_real_A.png','nicholas-prairie_real_A.png',
              'nicholas-beach_real_A.png','nicholas-field_fake_B.png','nicholas-prairie_fake_B.png',
              'nicholas-beach_fake_B.png']
    for i in imgs:
        imgs_root_path = os.path.join(path,
                                      'symbolism_nicholas/test_latest/images')
        img = Image.open(os.path.join(
                imgs_root_path,i)).convert('RGB')
        images.append(img)
        
    ori_img = ['field2.jpg', 'field1.png', 'beach.jpg']
    for o in ori_img:
        
        img = Image.open(os.path.join('data/testA',o)).convert('RGB')
        images.append(img)
        
    fig, ax = plt.subplots(3, 4, figsize=(20,15), constrained_layout=False)
    ax[0][0].imshow(images[0])
    ax[0][0].set_title('Style',fontsize=16)
    ax[0][1].imshow(images[1])
    ax[0][1].set_title('Nicholas - Field',fontsize=16)
    ax[0][2].imshow(images[2])
    ax[0][2].set_title('Nicholas - Prairie',fontsize=16)
    ax[0][3].imshow(images[3])
    ax[0][3].set_title('Nicholas - Beach',fontsize=16)
    ax[1][1].imshow(images[4])
    ax[1][1].set_title('Recovered - Field',fontsize=16)
    ax[1][2].imshow(images[5])
    ax[1][2].set_title('Recovered - Prairie',fontsize=16)
    ax[1][3].imshow(images[6])
    ax[1][3].set_title('Recovered - Beach',fontsize=16)
    ax[2][1].imshow(images[7])
    ax[2][1].set_title('Original - Field',fontsize=16)
    ax[2][2].imshow(images[8])
    ax[2][2].set_title('Original - Prairie',fontsize=16)
    ax[2][3].imshow(images[9])
    ax[2][3].set_title('Original - Beach',fontsize=16)
    
    for i in range(3):
        for j in range(4):
            ax[i][j].axis('off')
    plt.subplots_adjust(wspace=.001, hspace=.1)