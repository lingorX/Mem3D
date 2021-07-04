import scipy.misc as sm
import numpy as np
import timeit
import matplotlib.pyplot as plt
from collections import OrderedDict
# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
import dataloaders.pretrain_loader as pretrain_loader
import dataloaders.CVC as CVC
from dataloaders import custom_transforms2 as tr 
from dataloaders.helpers2 import *
from networks.loss import class_cross_entropy_loss
from networks.mainnetwork import *

# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print('Using GPU: {} '.format(gpu_id))

# Setting parameters
nEpochs = 600  # Number of epochs for training
resume_epoch = 351 # Default is 0, change if want to resume
p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 15  # Training batch size 5
snapshot = 10  # Store a model every snapshot epochs
nInputChannels = 5  # Number of input channels (RGB + heatmap of extreme points)
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
run_id = 2
save_dir = os.path.join(save_dir_root, 'two_' + str(run_id))
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
    print("Initializing from pretrained model")
else:
    print("Initializing weights from: {}".format(
        os.path.join('two_1', 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    pretrained_dict = torch.load(os.path.join('two_1', 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
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
    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image':(512,512),'gt': (512, 512)},flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, elem='gt',pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('image', 'IOG_points')),
        tr.ToTensor()])
                                            
    voc_train = CVC.CVCSegmentation(split='train', transform=composed_transforms_tr)
    db_train = voc_train

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=8, drop_last=True)

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
            gts =  sample_batched['gt']
            inputs = sample_batched['concat']
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
