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
        _mask_dir = os.path.join(_msd_root, 'Task06_mask')#each object each color
        _image_dir = os.path.join(_msd_root, 'Task06_origin')
        self.transform = transform
        self.split = split
        _splits_dir = os.path.join(_msd_root, 'ImageSets06')
        self.image_ids = []
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
        print('len : {}'.format(len(self.image_ids)))


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}
        sample['meta'] = {'image': str(self.image_ids[index]),
                          'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_ids)

    def _make_img_gt_point_pair(self, index):
        _msd_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_msd_root, 'Task06_mask')#each object each color
        _image_dir = os.path.join(_msd_root, 'Task06_origin')
        _img = np.array(Image.open(os.path.join(_image_dir, self.image_ids[index].split('-')[0], self.image_ids[index].split('-')[1])).convert('RGB')).astype(np.float32)
        _tmp = np.array(Image.open(os.path.join(_mask_dir, self.image_ids[index].split('-')[0], self.image_ids[index].split('-')[1]))).astype(np.float32)
        _tmp[_tmp!=1] = 0 
        _tmp[_tmp==1] = 255.0

        return _img, _tmp

    def __str__(self):
        return 'CVC-ClinicDB(split=' + str(self.split)

