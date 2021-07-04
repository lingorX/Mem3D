import torch
import os
import math
import cv2
import numpy as np
import torch

import json
import yaml
import random
import pickle

from PIL import Image
from torch.utils.data import Dataset

ROOT = 'data'

class CVCVOS(Dataset):

    def __init__(self,transform=None,whole= True):
        data_dir = os.path.join(ROOT, 'MSD')

        split = 'test'

        self.root = data_dir
        with open(os.path.join(data_dir, 'ImageSets10', split + '.txt'), "r") as lines:
            videos = []
            for line in lines:
                _video = line.rstrip('\n')
                videos.append(_video)
        self.videos = videos

        self.length = len(self.videos)
        self.whole = whole
        self.transform = transform

    def __getitem__(self, idx):

        vid = self.videos[idx]
        path = os.path.join(self.root, 'Task10_mask', vid)
        _mask_dir = os.path.join(self.root, 'Task10_mask')#each object each color
        _image_dir = os.path.join(self.root, 'Task10_origin')
        frames = os.listdir(path) 
        frames.sort(key=lambda x:int(x.split('.')[0]))
        if self.whole:
            framess = [np.array(Image.open(os.path.join(_image_dir, vid, name)).convert('RGB')) for name in frames]
            names = [vid+'-'+name for name in frames]
            # masks = [np.array(Image.open(os.path.join(_mask_dir, vid, name))) for name in frames]
            masks = []
            for item in frames:
                mask = np.array(Image.open(os.path.join(_mask_dir, vid, item)))
                mask[mask!=1]=0
                masks.append(mask)
        else:
            masks = []
            framess = []
            names = []
            for item in frames:
                mask = np.array(Image.open(os.path.join(_mask_dir, vid, item)))
                # if len(mask[mask==1])>0:
                if len(mask[mask==1])>0:
                    framess.append(np.array(Image.open(os.path.join(_image_dir, vid, item)).convert('RGB')))
                    # mask[mask!=1]=0
                    mask[mask!=1]=0
                    masks.append(mask)
                    names.append(vid+'-'+item)

        return framess, masks, names

    def __len__(self):
        
        return self.length
