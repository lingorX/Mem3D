import torch
import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import random
from mypath import Path

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

class CVCVideoSegmentation(data.Dataset):

    BASE_DIR = 'MSRA_10K'

    def __init__(self,
                 root=Path.db_root_dir('CVC'),
                 split='test',
                 transform=None,
                 #Instance segmentation disabled?
                 default=True):

        self.root = root
        _cvc_root = os.path.join(self.root, self.BASE_DIR)
        _image_dir = os.path.join(_cvc_root, 'Img_gt')
        self.transform = transform
        self.split = split
        self.default = default

        self.images = []
        self.masks = []

        with open(os.path.join(os.path.join(_cvc_root, 'ImageSets', 'train.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + ".jpg")
            _mask = os.path.join(_image_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.images.append(_image)
            self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):

        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['meta'] = {'image': str(self.images[index].split('/')[-1]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        return sample

    def _make_img_gt_point_pair(self, index):
 
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        maskk = np.array(Image.open(self.masks[index]))
        _tmp = maskk.astype(np.float32)
        _noise= (_tmp !=255 )
        _tmp[_noise] = 0 

        # fullfill other regions
        contours = cv2.findContours(
            image=_tmp.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
        masked_img=[]
        if(len(contours)>1):
            for idx,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>100):
                    newcontours = contours.copy()
                    del(newcontours[idx])
                    con = np.array(_tmp.astype(np.uint8),copy=True)
                    cv2.drawContours(con,newcontours,-1,(0,0,0),thickness=-1)
                    masked_img.append(con.astype(np.float32))

        # if(len(masked_img)>1):
        #     for i,con in enumerate(masked_img):
        #         cv2.imwrite('data/'+self.images[_im_ii].split('/')[-1].split('.')[0]+"_"+str(i)+".png",con)
            
        _target = _tmp
        if(len(masked_img)>1) and self.split=='train':
            _target = masked_img[random.sample(range(0, len(masked_img)), 1)[0]]
        elif len(masked_img)>1:
            _target = masked_img

        return _img, _target

    def __len__(self):
        return len(self.images)
