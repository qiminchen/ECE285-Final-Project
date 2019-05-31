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


# Dataset
class StyleTransferDataset(Dataset):
    def __init__(self, root_dir, img_type, image_size=(512, 512)):
        super(StyleTransferDataset, self).__init__()
        self.image_size = image_size
        self.images_dir = os.path.join(root_dir, img_type)
    
    
    def __getitem__(self, types, img_name):
        
        img_path = os.path.join(self.images_dir, types, img_name)
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

# content layer: ‘conv3 1’ with index 5
content_layers_idx = [5]

'''
conv1_1 with index 1
conv2_1 with index 3
conv3_1 with index 5
conv4_1 with index 9
conv5_1 with index 13
'''

style_layers_idx = [1, 3, 5, 9, 13]

def get_style_model_and_losses(net, style_img, content_img, 
                               content_layers=content_layers_idx, 
                               style_layers=style_layers_idx):
    
    vgg = copy.deepcopy(net)
    
    # losses
    content_losses = []
    style_losses = []
    
    model = nn.Sequential()
    
    i = 0
    for layer in vgg.children():
        
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)
        
        if i in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if i in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    
    
    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i + 1)]
    
    return model, style_losses, content_losses
    
    
    
    
    