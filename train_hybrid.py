from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import numpy as np
import timeit
# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
import dataloaders.CVC_hybrid as CVC
from dataloaders import custom_transforms_dextr as tr_dextr 
from dataloaders import custom_transforms as tr_box
import os
import cv2
import random
from networks.loss import class_cross_entropy_loss
from networks.mainnetwork import *

# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print('Using GPU: {} '.format(gpu_id))

# Setting parameters
use_sbd = False # train with SBD
nEpochs = 1000  # Number of epochs for training
resume_epoch = 520  # Default is 0, change if want to resume
p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 1  # Training batch size 5
snapshot = 10  # Store a model every snapshot epochs
nInputChannels = 5  # Number of input channels (RGB + heatmap of extreme points)
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

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
                        freeze_bn=False,
                        pretrained=True)

net = torch.nn.DataParallel(net).cuda()

if resume_epoch == 0:
    print("Pretain weights from: {}".format(
        os.path.join('two_1', 'models', modelName + '_epoch-319.pth')))

    pretrained_dict = torch.load(os.path.join('two_1', 'models', modelName + '_epoch-319.pth'))
    #model.load_state_dict(pretrained_dict)
    model_dict = net.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    conv1_weight_new=np.zeros( (64,nInputChannels,7,7) )
    conv1_weight_new[:,:3,:,:]=pretrained_dict['module.backbone.conv1.weight'].cpu().data[:,:3,:,:]
    pretrained_dict['module.backbone.conv1.weight']=torch.from_numpy(conv1_weight_new)
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))

    pretrained_dict = torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
    #model.load_state_dict(pretrained_dict)
    model_dict = net.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

train_params = [{'params': net.module.get_1x_lr_params(), 'lr': p['lr']}, 
                {'params': net.module.get_10x_lr_params(), 'lr': p['lr'] * 10}]



if resume_epoch != nEpochs:
    # Logging into Tensorboard

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_dextr = transforms.Compose([
        tr_dextr.RandomHorizontalFlip(),
        tr_dextr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr_dextr.CropFromMask(crop_elems=('image', 'gt'), relax=50, zero_pad=True),
        tr_dextr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)},flagvals={'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR}),
        tr_dextr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
        tr_dextr.ToImage(norm_elem='extreme_points'),
        tr_dextr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr_dextr.ToTensor()])
                 
    composed_transforms_box = transforms.Compose([
        tr_box.RandomHorizontalFlip(),
        tr_box.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr_box.CropFromMask(crop_elems=('image', 'gt'), relax=30, zero_pad=True),
        tr_box.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)},flagvals={'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR}),
        tr_box.IOGPoints(sigma=10, elem='crop_gt',pad_pixel=10),
        tr_box.ToImage(norm_elem='IOG_points'),
        tr_box.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr_box.ToTensor()])


    composed_transforms_scribble = transforms.Compose([
        tr_box.RandomHorizontalFlip(),
        tr_box.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr_box.CropFromMask(mask_elem='scribble', crop_elems=('image', 'gt', 'scribble'), relax=50, zero_pad=True),
        tr_box.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_scribble': (512, 512)},flagvals={'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR,'crop_scribble':cv2.INTER_LINEAR}),
        tr_box.ConcatInputs(elems=('crop_image', 'crop_scribble')),
        tr_box.ToTensor()])

    voc_train = CVC.CVCSegmentation(split='train', transform=[composed_transforms_dextr, composed_transforms_scribble, composed_transforms_box])
    db_train = voc_train

    p['dataset_train'] = str(db_train)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2,drop_last=True)

    # Train variables
    num_img_tr = len(trainloader)
    running_loss_tr = 0.0
    aveGrad = 0
    print("Training Network")
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        epoch_loss = []
        net.train()
        for ii, sample_batched in enumerate(trainloader):
            try:
                gts =  sample_batched['crop_gt'][0]
            except KeyError:
                gts =  sample_batched['gt'][0]
            inputs = sample_batched['concat'][0]
            inputs.requires_grad_()
            inputs, gts = inputs.cuda(), gts.cuda()
            coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)
            # Compute the losses
            loss_coarse_outs1 = class_cross_entropy_loss(coarse_outs1, gts, void_pixels=None)
            loss_coarse_outs2 = class_cross_entropy_loss(coarse_outs2, gts, void_pixels=None)
            loss_coarse_outs3 = class_cross_entropy_loss(coarse_outs3, gts, void_pixels=None)
            loss_coarse_outs4 = class_cross_entropy_loss(coarse_outs4, gts, void_pixels=None)
            loss_fine_out = class_cross_entropy_loss(fine_out, gts, void_pixels=None)
            loss = loss_coarse_outs1+loss_coarse_outs2+ loss_coarse_outs3+loss_coarse_outs4+loss_fine_out

            if ii % 10 ==0:
                print('Epoch',epoch,'step',ii,'loss',loss)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1 -p['trainBatch']:
                running_loss_tr = running_loss_tr / num_img_tr
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time)+"\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
