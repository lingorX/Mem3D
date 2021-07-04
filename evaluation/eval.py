import os.path
import cv2
import numpy as np
from PIL import Image

import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation

def eval_one_result(loader, folder, mask_thres=0.5,whole=False):
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["all_dice"] = np.zeros(len(loader))
    # Iterate
    for i, sample in enumerate(loader):

        frames, masks, names = sample
        _mask = []
        _gt = []
        for ii in range(len(masks)):
            filename = os.path.join(folder,names[ii][0])
            mask = np.array(Image.open(filename)).astype(np.float32)
            gt = np.squeeze(helpers.tens2image(masks[ii]))
            mask = (mask > mask_thres)
            _mask.append(mask)
            _gt.append(gt)
        eval_result["all_jaccards"][i] = evaluation.jaccard(np.stack(_gt,0), np.stack(_mask,0))
        eval_result["all_dice"][i] = evaluation.dice(np.stack(_gt,0), np.stack(_mask,0))

    return eval_result



