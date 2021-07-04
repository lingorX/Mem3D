import torch
import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
from six.moves import urllib
import json
from mypath import Path
import random

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

class CVCSegmentation(data.Dataset):

    BASE_DIR = 'MSD'

    def __init__(self,
                 root=Path.db_root_dir('MSD'),
                 split='test',
                 transform=None):
        self.root = root
        _msd_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_msd_root, 'Task10_mask')#each object each color
        _scribble_dir = os.path.join(_msd_root, 'Task10_mask2')
        _image_dir = os.path.join(_msd_root, 'Task10_origin')
        self.transform = transform
        self.split = split
        _splits_dir = os.path.join(_msd_root, 'ImageSets10')
        self.image_ids = []
        self.scribble_ids = []
        with open(os.path.join(_splits_dir, split + '.txt'), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            path = os.path.join(_mask_dir, line.rstrip('\n'))
            filelist = os.listdir(path) 
            for i in range(0, len(filelist)):
                _mask = os.path.join(_mask_dir, line.rstrip('\n'), filelist[i])
                assert os.path.isfile(_mask)
                maskk = np.array(Image.open(_mask)).astype(np.float32)
                if len(maskk[maskk==1])>0:
                    self.image_ids.append(line.rstrip('\n') + '-' + filelist[i])

        for ii, line in enumerate(lines):
            path = os.path.join(_scribble_dir, line.rstrip('\n'))
            if not os.path.exists(path):
                continue
            filelist = os.listdir(path) 
            for i in range(0, len(filelist)):
                _scribble = os.path.join(_scribble_dir, line.rstrip('\n'), filelist[i])
                assert os.path.isfile(_scribble)
                self.scribble_ids.append(line.rstrip('\n') + '-' + filelist[i])

        print('len : {}'.format(len(self.image_ids)))


    def __getitem__(self, index):
        chance = random.sample(range(0, 4), 1)[0]

        sample_batched = []
        image_ids = []
        for ii in range(index*5,index*5+5):
            if chance==1:
                _img, _target,_scribble = self._make_img_gt_scribble_pair()
                sample = {'image': _img, 'gt': _target,'scribble':_scribble}
            else:
                _img, _target = self._make_img_gt_point_pair(ii)
                sample = {'image': _img, 'gt': _target}
            image_ids.append(str(self.image_ids[ii]))

            if self.transform is not None:
                if chance==0:
                    sample = self.transform[0](sample)
                    sample['concat'] = torch.cat((sample['concat'], torch.zeros([1,512,512])),0)
                elif chance==1:
                    sample = self.transform[1](sample)
                    sample['concat'] = torch.cat((sample['concat'], torch.zeros([1,512,512])),0)
                elif chance==2:
                    sample = self.transform[2](sample)
                elif chance==3:
                    sample = self.transform[2](sample)
                    sample['concat'] = torch.cat((sample['concat'][0:3], sample['concat'][4:5]),0)
                    sample['concat'] = torch.cat((sample['concat'], torch.zeros([1,512,512])),0)

            sample_batched.append(sample)
        data={'meta':{}}
        for key in sample_batched[0].keys():
            data[key] = torch.stack((sample_batched[0][key],sample_batched[1][key],sample_batched[2][key],sample_batched[3][key],sample_batched[4][key]),dim=0)

        data['meta']['image'] = image_ids
        return data

    def __len__(self):
        return int(len(self.image_ids)/5)

    def _make_img_gt_point_pair(self, index):
        _msd_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_msd_root, 'Task10_mask')#each object each color
        _image_dir = os.path.join(_msd_root, 'Task10_origin')
        _img = np.array(Image.open(os.path.join(_image_dir, self.image_ids[index].split('-')[0], self.image_ids[index].split('-')[1])).convert('RGB')).astype(np.float32)
        _tmp = np.array(Image.open(os.path.join(_mask_dir, self.image_ids[index].split('-')[0], self.image_ids[index].split('-')[1]))).astype(np.float32)
        _tmp[_tmp!=1] = 0 
        _tmp[_tmp==1] = 255.0

        return _img, _tmp

    def _make_img_gt_scribble_pair(self):
        index = random.sample(range(0, len(self.scribble_ids)), 1)[0]
        _msd_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_msd_root, 'Task10_mask')#each object each color
        _scribble_dir = os.path.join(_msd_root, 'Task10_mask2')
        _image_dir = os.path.join(_msd_root, 'Task10_origin')
        _img = np.array(Image.open(os.path.join(_image_dir, self.scribble_ids[index].split('-')[0], self.scribble_ids[index].split('-')[1])).convert('RGB')).astype(np.float32)
        _tmp = np.array(Image.open(os.path.join(_mask_dir, self.scribble_ids[index].split('-')[0], self.scribble_ids[index].split('-')[1]))).astype(np.float32)
        _scribble = np.array(Image.open(os.path.join(_scribble_dir, self.scribble_ids[index].split('-')[0], self.scribble_ids[index].split('-')[1]))).astype(np.float32) 
        _tmp[_tmp!=1] = 0 
        _tmp[_tmp==1] = 255.0

        return _img, _tmp, _scribble

    def __str__(self):
        return 'CVC-ClinicDB(split=' + str(self.split)

