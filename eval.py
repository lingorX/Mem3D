import os.path

from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.eval_loader as eval_loader
import numpy as np

exp_root_dir = './'

method_names = []
method_names.append('')

if __name__ == '__main__':

    # Dataloader
    dataset = eval_loader.CVCVOS(transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Iterate through all the different methods
    for method in method_names:
        for ii in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            results_folder = os.path.join(exp_root_dir, method, 'Results2')
    
            jaccards = eval_one_result(dataloader, results_folder, mask_thres=ii)
            val = jaccards["all_jaccards"].mean()
            val2 = jaccards["all_dice"].mean()
            # Show mean and store result
            print(ii)
            print(jaccards['all_jaccards'])
            print(jaccards['all_dice'])
            print('Standard Deviation {}'.format(np.std(jaccards['all_dice'], ddof=1)))
            print("IoU Result for {:<80}: {}".format(method, str.format("{0:.4f}", 100*val)))
            print("Dice Result for {:<80}: {}".format(method, str.format("{0:.4f}", 100*val2)))
