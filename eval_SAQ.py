from dataloaders.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from dataloaders.transform import TrainTransform, TestTransform
from utils.logger import Logger, AverageMeter
from utils.loss import *
from utils.utility import write_mask, write_mask2, save_checkpoint, adjust_learning_rate, mask_iou
import evaluation.evaluation as evaluation
from models_STM.models_SAQ import STM
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import time
from progress.bar import Bar
from options import OPTION as opt

# Use CUDA
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

####IOG START########IOG START########IOG START########IOG START########IOG START####
import scipy.misc as sm
import numpy as np
# PyTorch includes
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
# Custom includes
from dataloaders import custom_transforms_dextr as tr
from dataloaders.helpers_dextr import *
from networks.mainnetwork import *

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
composed_transforms_IOG = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt'), relax=50, zero_pad=True),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)},flagvals={'gt':cv2.INTER_LINEAR,'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR}),
    tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
    tr.ToImage(norm_elem='extreme_points'),
    tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
    tr.ToTensor()])

def load_trained_network(model, path):
    print('load model from : {}'.format(path))
    model = torch.nn.DataParallel(model).cuda()
    pretrain_dict = torch.load(path,map_location='cpu')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model
def make_img_gt_point_pair_IOG(frame_name):
    _img = np.array(Image.open('data/MSD/Task06_origin/'+frame_name).convert('RGB')).astype(np.float32) ###zsy open image imread 
    # Read Target object
    maskk = np.array(Image.open('data/MSD/Task06_mask/'+frame_name))
    _tmp = maskk.astype(np.float32)
    _tmp[_tmp>0]=255.0
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
                if con.max()>0:
                    masked_img.append(con.astype(np.float32))

    _target = _tmp
    if len(masked_img)>1:
        _target = masked_img

    return _img, _target
####IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED####

def main():

    ####IOG START########IOG START########IOG START########IOG START########IOG START####
    resume_epoch_IOG = 200
    nInputChannels_IOG = 4
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    run_id = 12
    save_dir_IOG = os.path.join(save_dir_root, 'run_' + str(run_id))
    net_IOG = Network(nInputChannels=nInputChannels_IOG,num_classes=1,
                    backbone='resnet101',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    net_IOG = load_trained_network(net_IOG, os.path.join(save_dir_IOG, 'models', 'IOG_CVC_epoch-' + str(resume_epoch_IOG - 1) + '.pth'))
    net_IOG.eval()
    ####IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED####

    print('==> Preparing dataset %s' % opt.valset)
    input_dim = opt.input_size
    test_transformer = TestTransform(size=input_dim)
    testset = DATA_CONTAINER[opt.valset](
        train=False, 
        transform=test_transformer, 
        samples_per_video=1
        )
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    print("==> creating model")
    net = STM(opt.keydim, opt.valdim)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    net.eval()
    if use_gpu:
        net.to(device)
    for p in net.parameters():
        p.requires_grad = False
    title = 'STM'
    if opt.resume:
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume, map_location=device)
        state = checkpoint['state_dict']
        net.load_param(state)
    print('==> Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))
    test(testloader,
        model=net,
        net_IOG=net_IOG,
        use_cuda=use_gpu,
        opt=opt)
    print('==> Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))

def test(testloader, model, net_IOG, use_cuda, opt):
    criterion = mask_iou_loss
    data_time = AverageMeter()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            frames, masks, objs, infos = data
            if use_cuda:
                frames = frames.to(device)
                masks = masks.to(device)
            frames = frames[0]
            masks = masks[0]
            num_objects = objs[0]
            info = infos[0]
            max_obj = masks.shape[1]-1
            T, _, H, W = frames.shape

            ####IOG START########IOG START########IOG START########IOG START########IOG START####
            ref_name = info['frame_name'][0]
            print('==>' + ref_name + ' single slice as reference')
            fine_out = interactive_seg(net_IOG, info, ref_name)
            ####IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED########IOG ENDED####

            ####STM START########STM START########STM START########STM START########STM START####
            print('==>Runing video {}, objects {:d}'.format(info['name'], num_objects))
            frame_certainty_map={}
            pred, frame_certainty_map = STM_inference(frames, model, info, fine_out, T, frame_certainty_map, num_objects, max_obj, data_time)
            write_mask(pred, info, opt, directory=opt.output_dir)
            ####STM START########STM START########STM START########STM START########STM START####

            print('==> Refinement starting...')
            poor_list = [ref_name]
            origin_frames = sorted(info['frame_name'],key=lambda x:int(x.split('.')[0]))
            for ij in range(0, opt.refine_time):
                print('round: '+ str(ij+1))
                
                # uncertainty-based
                # cer_list = sorted(frame_certainty_map.items(), key=lambda item:item[1])
                # print(cer_list)
                # for ii in range(len(cer_list)):
                #     gt = np.array(Image.open('data/MSD/Task06_mask/'+ info['name']+'/'+cer_list[ii][0])).astype(np.float32)
                #     gt[gt>0]=1
                #     if len(gt[gt==1])<100:
                #       continue
                #     else:
                #         poor_name = cer_list[ii][0]
                #         print('==>' + os.path.join(info['name'], cer_list[ii][0]) + ' is ready to be refined')
                #         break

                # gt-based
                poor_name = select_poorest_mask(opt.output_dir, info['name'])

                try:
                    poor_list.index(poor_name)
                except ValueError:
                    poor_list.append(poor_name)
                poor_list.sort(key=lambda x:int(x.split('.')[0]))
                posi = poor_list.index(poor_name)
                left = 0
                right = len(origin_frames)
                if posi == 0:
                    if len(poor_list)==1:
                        right = len(origin_frames)
                    else:
                        right = origin_frames.index(poor_list[1])
                elif posi == len(poor_list)-1:
                    left = origin_frames.index(poor_list[posi-1]) + 1
                else:
                    left = origin_frames.index(poor_list[posi-1]) + 1
                    right = origin_frames.index(poor_list[posi+1])
                print('start frame: ' + origin_frames[left] + '; end frame: ' + origin_frames[right-1])
                new_frame_list = origin_frames[left:right]
                new_ref_index = new_frame_list.index(poor_name)
                sample_frames = []
                sample_frames.append(new_frame_list[new_ref_index])
                for ii in range(1, max(new_ref_index+1, len(new_frame_list)-new_ref_index)):
                    if new_ref_index-ii>=0:
                        sample_frames.append(new_frame_list[new_ref_index-ii])
                    if new_ref_index+ii<len(new_frame_list):
                        sample_frames.append(new_frame_list[new_ref_index+ii])
                new_frame_list = sample_frames[0:min(5, len(sample_frames))]

                ####refine START########refine START########refine START########refine START########refine START####
                ref_name = new_frame_list[0]
                fine_out = interactive_seg(net_IOG, info, ref_name)
                frame_certainty_map[new_frame_list[0]]=1.0
                ####refine ENDED########refine ENDED########refine ENDED########refine ENDED########refine ENDED####

                new_frames = []
                for t in range(0, len(new_frame_list)):
                    index = info['frame_name'].index(new_frame_list[t])
                    new_frames.append(frames[index])
                new_frames = torch.stack(new_frames,0)

                ####STM START########STM START########STM START########STM START########STM START####
                pred, frame_certainty_map = STM_inference(new_frames, model, info, fine_out, len(new_frame_list), frame_certainty_map, num_objects, max_obj, data_time)
                ####STM START########STM START########STM START########STM START########STM START####

                write_mask2(pred, info, new_frame_list,opt, directory=opt.output_dir)
                print('round: '+ str(ij+1) + ' ended\n')
    return



def interactive_seg(net_IOG, info, ref_name):
    _img, _target = make_img_gt_point_pair_IOG(info['name']+'/'+ref_name)
    if isinstance(_target,list):
        sample = []
        for idx,con in enumerate(_target):
            item = {'image': _img, 'gt': con}
            item['meta'] = {'image':str(ref_name),'im_size': (_img.shape[0], _img.shape[1])}
            item = composed_transforms_IOG(item)
            sample.append(item)
    else:
        sample = {'image': _img, 'gt': _target}
        sample['meta'] = {'image': str(ref_name),'im_size': (_img.shape[0], _img.shape[1])}
        sample = composed_transforms_IOG(sample)

    if isinstance(sample,list):
        results = []
        for idx,con in enumerate(sample):
            ref_image, gts, metas = con['concat'].unsqueeze(0), con['gt'].unsqueeze(0), con['meta']
            ref_image = ref_image.cuda()
            coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net_IOG.forward(ref_image)
            outputs = fine_out.to(torch.device('cpu'))
            pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            gt = tens2image(gts[0, :, :, :])
            bbox = get_bbox(gt, pad=50, zero_pad=True)
            result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)
            results.append(result)
        fine_out = np.sum(np.array(results),axis=0)
    else:
        ref_image, gts, metas = sample['concat'].unsqueeze(0), sample['gt'].unsqueeze(0), sample['meta']
        ref_image = ref_image.cuda()
        coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net_IOG.forward(ref_image)
        outputs = fine_out.to(torch.device('cpu'))
        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        gt = tens2image(gts[0, :, :, :])
        bbox = get_bbox(gt, pad=50, zero_pad=True)
        fine_out = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)

    fine_out[fine_out >= 0.8] = 1
    fine_out[fine_out < 0.8] = 0
    oh = []
    for k in range(2):
        oh.append(fine_out==k)
    fine_out = np.stack(oh, axis=0)
    fine_out = torch.from_numpy(fine_out.astype(np.float32)).unsqueeze(0).to(device)

    return fine_out

def STM_inference(frames, model, info, fine_out, T, frame_certainty_map, num_objects, max_obj, data_time):
    bar = Bar(info['name'], max=T-1)
    # compute output
    pred = [fine_out[0:1]]
    keys = []
    vals = []
    for t in range(1, T):
        if t-1 == 0:
            tmp_mask = fine_out[0:1]
        else:
            tmp_mask = out
        t1 = time.time()
        # memorize
        key, val, _ = model(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)
        # segment
        tmp_key = torch.cat(keys+[key], dim=1)
        tmp_val = torch.cat(vals+[val], dim=1)
        logits, ps, quality = model(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj)
        out = torch.softmax(logits, dim=1)

        ####Quality Assess########Quality Assess########Quality Assess########Quality Assess########Quality Assess####
        cer_flag = out[0][1].clone().detach()
        cer_flag[cer_flag<=0.95]=0
        cer_flag[cer_flag>0.95]=1
        numbers_cer = cer_flag.sum(dim=[0,1])
        if numbers_cer>100:
            frame_certainty_map[info['frame_name'][t]]=quality.squeeze().item()
        else:
            frame_certainty_map[info['frame_name'][t]]=1.0
        ####Quality Assess########Quality Assess########Quality Assess########Quality Assess########Quality Assess####

        pred.append(out)
        if (t-1) % opt.save_freq == 0:
            keys.append(key)
            vals.append(val)
        # _, idx = torch.max(out, dim=1)
        toc = time.time() - t1
        data_time.update(toc, 1)

        # plot progress
        bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
            batch=t,
            size=T-1,
            data=data_time.sum
        )
        bar.next()
    bar.finish()
    pred = torch.cat(pred, dim=0)
    pred = pred.detach().cpu().numpy()

    return pred, frame_certainty_map

def select_poorest_mask(output_dir, name):
    path = os.path.join(ROOT, output_dir, opt.valset, name)
    filelist = os.listdir(path) 
    jaccards = []
    for i in range(0, len(filelist)):
        mask = np.array(Image.open(os.path.join(path, filelist[i]))).astype(np.float32)
        gt = np.array(Image.open('data/MSD/Task06_mask/'+ name+'/'+filelist[i].split('-')[1])).astype(np.float32)
        gt[gt>0]=1
        if len(gt[gt==1])<100:
            jaccards.append(1.0)
        else:
            jaccards.append(evaluation.jaccard(gt, mask))
    # print(jaccards)
    poor_idx = jaccards.index(min(jaccards))
    print('==>' + os.path.join(path, filelist[poor_idx]) + ' is ready to be refined')
    return filelist[poor_idx].split('-')[1]

if __name__ == '__main__':
    main()
