from datetime import datetime
import scipy.misc as sm
import glob
import numpy as np

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
# Custom includes
import dataloaders.CVC as CVC
from dataloaders import custom_transforms as tr
from dataloaders.helpers import *
from networks.mainnetwork import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setting parameters
resume_epoch = 140
nInputChannels = 5

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
run_id = 11
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'IOG_CVC'
net = Network(nInputChannels=nInputChannels,num_classes=1,
                backbone='resnet101',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)

net = torch.nn.DataParallel(net).cuda()
# load pretrain_dict
print(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
pretrain_dict = torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),map_location='cpu')
model_dict = net.state_dict()
pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict.keys()}
model_dict.update(pretrain_dict)
net.load_state_dict(model_dict)


# Generate result of the validation images
net.eval()
composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt'), relax=30, zero_pad=True),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)},flagvals={'gt':cv2.INTER_LINEAR,'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR}),
    tr.IOGPoints(sigma=10, elem='crop_gt',pad_pixel=10),
    tr.ToImage(norm_elem='IOG_points'),
    tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
    tr.ToTensor()])
db_test = CVC.CVCSegmentation(split='test', transform=composed_transforms_ts)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

save_dir_res = os.path.join(save_dir, 'Results')
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)
save_dir_inter = os.path.join(save_dir, 'Inter')
if not os.path.exists(save_dir_inter):
    os.makedirs(save_dir_inter)
save_dir_res_list=[save_dir_res, save_dir_inter]
print('Testing Network')
with torch.no_grad():
    for ii, sample_batched in enumerate(testloader):  
        if isinstance(sample_batched,list):
            results = []
            for idx,con in enumerate(sample_batched):
                inputs, gts, metas = torch.cat((con['concat'][:,0:3], con['concat'][:,4:5], torch.zeros_like(con['concat'][:,4:5])),1), con['gt'], con['meta']
                inputs = inputs.cuda()
                coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)
                outputs = fine_out.to(torch.device('cpu'))
                pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)
                gt = tens2image(gts[0, :, :, :])
                bbox = get_bbox(gt, pad=30, zero_pad=True)
                result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)
                results.append(result)
            sm.imsave(os.path.join(save_dir_res_list[0], sample_batched[0]['meta']['image'][0]),np.sum(np.array(results),axis=0))
        else:
            inputs, gts, metas = torch.cat((sample_batched['concat'][:,0:3], sample_batched['concat'][:,4:5],torch.zeros_like(sample_batched['concat'][:,4:5])),1), sample_batched['gt'], sample_batched['meta']
            inputs = inputs.cuda()
            coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)
            outputs = fine_out.to(torch.device('cpu'))
            pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            gt = tens2image(gts[0, :, :, :])
            bbox = get_bbox(gt, pad=30, zero_pad=True)
            result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)

            _image_dir = os.path.join('./data/MSD', 'Task06_origin')
            _img = np.array(Image.open(os.path.join(_image_dir, sample_batched['meta']['image'][0].split('-')[0], sample_batched['meta']['image'][0].split('-')[1])).convert('RGB')).astype(np.float32)
            cv2.rectangle(_img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 215, 0), 4)
            sm.imsave(os.path.join(save_dir_res_list[1],  metas['image'][0]), np.rot90(_img))

            sm.imsave(os.path.join(save_dir_res_list[0],  metas['image'][0]), result)
