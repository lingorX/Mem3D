import torch
import os
import math
import cv2
import numpy as np
import torch

import json
import random
import pickle

from PIL import Image
from torch.utils.data import Dataset

DATA_CONTAINER = {}
ROOT = 'data'
MAX_TRAINING_OBJ = 1
MAX_TRAINING_SKIP = 5

def multibatch_collate_fn(batch):

    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])

    objs = [torch.LongTensor([sample[2]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[3] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, objs, info

def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    oh = np.stack(oh, axis=2)

    return oh

def convert_one_hot(oh, max_obj):

    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k

    return mask

class BaseData(Dataset):

    def increase_max_skip(self):
        pass

    def set_max_skip(self):
        pass

class CVCVOS(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=2, increment=1, samples_per_video=12):
        data_dir = os.path.join(ROOT, 'MSD')

        split = 'train' if train else 'test'

        self.root = data_dir
        self.part_frames = []
        self.none_frames = []
        with open(os.path.join(data_dir, 'ImageSets10', split + '.txt'), "r") as lines:
            videos = []
            for line in lines:
                _video = line.rstrip('\n')
                videos.append(_video)
                path = os.path.join(self.root, 'Task10_mask', line.rstrip('\n'))
                _mask_dir = os.path.join(self.root, 'Task10_mask')#each object each color
                frames = os.listdir(path) 

                part_frames = []
                none_frames = []
                for i in range(0, len(frames)):
                    _mask = os.path.join(_mask_dir, line.rstrip('\n'), frames[i])
                    assert os.path.isfile(_mask)
                    maskk = np.array(Image.open(_mask)).astype(np.float32)
                    if len(maskk[maskk==1]) > 0:
                        part_frames.append(frames[i])
                    else:
                        none_frames.append(frames[i])
                self.part_frames.append(part_frames)
                self.none_frames.append(none_frames)
        self.videos = videos

        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames

        self.length = len(self.videos) * samples_per_video
        self.max_obj = 1

        self.transform = transform
        self.train = train
        self.max_skip = max_skip
        self.increment = increment

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP) 

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip 

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]
        # path = os.path.join(self.root, 'Task10_mask', vid)
        # _mask_dir = os.path.join(self.root, 'Task10_mask')
        # _image_dir = os.path.join(self.root, 'Task10_origin')
        path = os.path.join(self.root, 'Task10_mask', vid)
        _mask_dir = os.path.join(self.root, 'Task10_mask')
        _image_dir = os.path.join(self.root, 'Task10_origin')
        frames = os.listdir(path) 
        part_frames = self.part_frames[(idx // self.samples_per_video)]
        none_frames = self.none_frames[(idx // self.samples_per_video)]
        whole_frames = frames
        whole_frames.sort(key=lambda x:int(x.split('.')[0]))
        frames = part_frames
        frames.sort(key=lambda x:int(x.split('.')[0]))
        nframes = len(frames)
        num_obj = 0
        info = {'name': vid}
        while num_obj == 0:
            while num_obj == 0:
                if self.train:
                    last_sample = -1
                    sample_frame = []
                    # none_index = random.sample(range(0, len(none_frames)), 1)[0]
                    # sample_frame.append(none_frames[none_index])
                    nsamples = min(self.sampled_frames, nframes)
                    for i in range(nsamples):
                        if i == 0:
                            last_sample = random.sample(range(0, nframes-nsamples+1), 1)[0]
                        else:
                            last_sample = random.sample(
                                range(last_sample+1, min(last_sample+self.max_skip+1, nframes-nsamples+i+1)), 1)[0]
                        sample_frame.append(frames[last_sample])

                    # none_index = random.sample(range(0, len(none_frames)), 1)[0]
                    # sample_frame.append(none_frames[none_index])
                else:
                    sample_frame = whole_frames
                if self.train:
                    ref_index = random.sample(range(0, len(sample_frame)), 1)[0]
                else:
                    ran = random.sample(range(-2, 3), 1)[0]
                    ref_index = whole_frames.index(frames[int(len(frames)/2+ran)])
                if self.train:
                    t = None
                    for ii in range(0, int(ref_index/2)+1):
                        t = sample_frame[ii]
                        sample_frame[ii] = sample_frame[ref_index - ii]
                        sample_frame[ref_index-ii] = t
                else:
                    sample_frames = []
                    sample_frames.append(sample_frame[ref_index])
                    for ii in range(1, max(ref_index+1, len(sample_frame)-ref_index)):
                        if ref_index-ii>=0:
                            sample_frames.append(sample_frame[ref_index-ii])
                        if ref_index+ii<len(sample_frame):
                            sample_frames.append(sample_frame[ref_index+ii])
                    sample_frame = sample_frames
                frame = [np.array(Image.open(os.path.join(_image_dir, vid, name)).convert('RGB')) for name in sample_frame]
                info['frame_name'] = [name for name in sample_frame]
                mask = []
                for name in sample_frame:
                    _m = np.array(Image.open(os.path.join(_mask_dir, vid, name))) 
                    _m[_m!=1]=0
                    # _m[_m!=2]=0
                    # _m[_m==2]=1
                    mask.append(_m)
                num_obj = int(mask[0].max())
            mask = [convert_mask(msk, self.max_obj) for msk in mask]

            info['palette'] = [0,0,0,255,255,255]
            info['size'] = frame[0].shape[:2]
            if self.transform is None:
                raise RuntimeError('Lack of proper transformation')
            frame, mask = self.transform(frame, mask, False)

            if self.train:
                num_obj = 0
                for i in range(1, MAX_TRAINING_OBJ+1):
                    if torch.sum(mask[0, i]) > 0:
                        num_obj += 1
                    else:
                        break
        return frame, mask, num_obj, info

    def __len__(self):
        
        return self.length

DATA_CONTAINER['MSD'] = CVCVOS
