#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:20:41 2019

@author: qiminchen
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import copy
import pandas as pd
import os

# dataset root directory
dataset_root_dir = '/datasets/ee285f-public/wikiart/'

# Dataset
class WikiArt(Dataset):
    def __init__(self, root_dir, mode="train", image_size=(512, 512)):
        super(WikiArt, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.data = pd.read_csv(os.path.join(root_dir, "Style/style_%s.csv" % mode))
        self.images_dir = os.path.join(root_dir, "wikiart")
    
    def __len__(self):
        
        return len(self.data)
    
    def __repr__(self):
        
        return "WikiArt(mode={}, image_size={})". \
                format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.images_dir, \
                   self.data.iloc[idx][0])
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
        
        img = transform(img)
    
        return img
    
# Content Loss
class ContentLoss(nn.Module):
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # 'detach' the target content from the tree used
        # to dynamically compute the gradient
        self.target = target.detach()
        
    def forward(self, input):
        
        self.loss = F.mse_loss(input, self.target)
        
        return input
    
# Style Loss
def gram_matrix(input):
    
    # a: batch size
    # b: number of feature maps
    # (c,d): dimensions of a feature map
    
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    
    # Normalize the values of the gram matrix
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        
        return input
    
    
# Model
vgg19 = models.vgg19(pretrained=True).features.eval()

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    
    
    
    
    
    
    
    
    
    