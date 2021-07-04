import os.path
import cv2
import numpy as np
from PIL import Image

import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation

def eval_one_result(loader, folder, mask_thres=0.5):
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["all_dice"] = np.zeros(len(loader))
    # Iterate
    for i, sample in enumerate(loader):

        frames, masks, names = sample
        for ii in range(len(masks)):
            filename = os.path.join(folder,names[ii][0])
            mask = np.array(Image.open(filename)).astype(np.float32)/255
            gt = np.squeeze(helpers.tens2image(masks[ii]))
            mask = (mask > mask_thres)
            eval_result["all_jaccards"][i] += evaluation.jaccard(gt, mask)
            eval_result["all_dice"][i] += evaluation.dice(gt, mask)
        eval_result['all_jaccards'][i] /= len(masks)
        eval_result['all_dice'][i] /= len(masks)

    return eval_result



