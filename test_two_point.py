from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import numpy as np

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
import dataloaders.CVC as CVC
from dataloaders import custom_transforms2 as tr
from networks.loss import class_cross_entropy_loss 
from dataloaders.helpers2 import *
from networks.mainnetwork import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
# gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print('Using GPU: {} '.format(gpu_id))

# Setting parameters
resume_epoch = 500 # test epoch
nInputChannels = 5  # Number of input channels (RGB + heatmap of IOG points)

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
run_id = 30
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
    tr.FixedResize(resolutions={'image':(512,512),'gt': (512, 512)},flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
    tr.IOGPoints(sigma=10, elem='gt',pad_pixel=10),
    tr.ToImage(norm_elem='IOG_points'),
    tr.ConcatInputs(elems=('image', 'IOG_points')),
    tr.ToTensor()])

db_test = CVC.CVCSegmentation(split='test', transform=composed_transforms_ts)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

save_dir_res = os.path.join(save_dir, 'Results')
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)
save_dir_res_list=[save_dir_res]
print('Testing Network')
with torch.no_grad():
    for ii, sample_batched in enumerate(testloader):  
        if isinstance(sample_batched,list):
            results = []
            for idx,con in enumerate(sample_batched):
                inputs, gts, metas = con['concat'], con['gt'], con['meta']
                inputs = inputs.cuda()
                coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)
                outputs = fine_out.to(torch.device('cpu'))
                pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)
                gt = tens2image(gts[0, :, :, :])
                bbox = get_bbox(gt, pad=30, zero_pad=True)
                result = cv2.resize(pred, (con['meta']['im_size'][1].item(),con['meta']['im_size'][0].item()), interpolation=cv2.INTER_CUBIC)
                results.append(result)
            print(sample_batched[0]['meta']['image'][0]+' fusion')
            sm.imsave(os.path.join(save_dir_res_list[0], sample_batched[0]['meta']['image'][0]),np.sum(np.array(results),axis=0))
        else:
            inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']
            inputs = inputs.cuda()
            coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)
            outputs = fine_out.to(torch.device('cpu'))
            pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            result = cv2.resize(pred, (sample_batched['meta']['im_size'][1].item(),sample_batched['meta']['im_size'][0].item()), interpolation=cv2.INTER_CUBIC)
            sm.imsave(os.path.join(save_dir_res_list[0], metas['image'][0]), result)
