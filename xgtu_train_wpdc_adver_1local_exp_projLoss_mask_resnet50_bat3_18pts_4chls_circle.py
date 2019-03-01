#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging
import os, cv2
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import scipy.io as sio
import matplotlib.pyplot as plt

from ddfa_utils2 import DDFADataset, ToTensorGjz, NormalizeGjz
from ddfa_utils2 import str2bool, AverageMeter
from io_utils import mkdir
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
import matplotlib.pyplot as plt
#from advers_proj_utils_2 import Discriminator, GenrVertex2lms, convert_to_ori, get_lms_image, get_lms_crop_pts
from advers_proj_utils_3 import *
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor
from resnet_xgtu_4chls import resnet50
from _4chls_utils_18LmsAid import *

# global (configuration)
arch="mobilenet_1" 
start_epoch=1 
param_fp_train='./train.configs/param_all_norm_decon.pkl'
param_fp_val='./train.configs/param_all_norm_val.pkl'
lms_vfp_train = './train.configs/trainData_2dPts_frmFAN.txt'
lms_vfp_val = './train.configs/lms_18pts_val.txt'
lmsImg_train='./UMDFaces/umdfaces_batch3_crop/'
lmsImg_file_path='./UMDFaces/umdfaces_batch3_crop/Landmarks_forCropFaces_frmFAN.txt'
warmup=-1 
opt_style='resample'
batch_size=8
base_lr=0.004
lr = base_lr
momentum = 0.9
weight_decay = 5e-4
epochs=50
#milestones=30,40,50,60
print_freq=50 
devices_id=[0]
workers=8 
filelists_train="./train.configs/train_aug_120x120_list_decon.txt"
filelists_val="./train.configs/train_aug_120x120.list.val" 
root="./train_aug_120x120" 
log_file="./train_results/wpdc_adverse_1local_exp_projLoss_mask_preTrn_bat3_18pts_4chls_circle/"
loss='wpdc'
snapshot="./train_results/wpdc_adverse_1local_exp_projLoss_mask_preTrn_bat3_18pts_4chls_circle/"
log_mode = 'w'
resume = ''
size_average = True
num_classes = 62
frozen = 'false'
task = 'all'
test_initial = False
resample_num = 132
cuda = True
#checkpoint_fp = './training_debug/2_wpdc_adverse_1local_exp_projLoss_18pts_mask_shufTrn/1_good/_checkpoint_epoch_60.pth.tar'

mkdir(snapshot)

_18_indx_3d22d = [17,19,21,22,24,26,36,40,39,42,46,45,31,30,35,48,66,54]

###################################### added by xgtu ###########################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

adversarial_loss = torch.nn.BCELoss()
# Initialize discriminator
discriminator_global = Discriminator(50)
discriminator_sh = Discriminator(40)
discriminator_exp = Discriminator(10)

if cuda:
    discriminator_global.cuda()
    discriminator_sh.cuda()
    discriminator_exp.cuda()
    adversarial_loss.cuda()

# Initialize weights
discriminator_global.apply(weights_init_normal)
optimizer_D = torch.optim.Adam(discriminator_global.parameters(), lr=0.00001, betas=(0.5, 0.999))
####################################### added by xgtu ##########################################

def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= warmup:
            return 1
        elif warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr
    lr = base_lr
    for param_group in optimizer.param_groups:
        lr = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def train(train_loader, model, criterion, optimizer, epoch, Loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()

    ff = open(lmsImg_file_path)
    lms_files = ff.readlines()
    #    lms_files = lms_files[1:]
    ff.close()
    idx = 0
    batch_idxs = len(lms_files) // batch_size
    tri = sio.loadmat('visualize/tri.mat')['tri']
#    tri = _numpy_to_cuda(tri.astype(np.float32))
    for i, (input, target, lms_list) in enumerate(train_loader):
#        print(i)
        lmsMap = [obtain_map(aa) for aa in lms_list]
        comInput1 = comb_inputs(input, lmsMap, permu=True)
        comInput1 = _numpy_to_cuda(comInput1)
        comInput1 = comInput1.permute(0, 3, 1, 2)

        batLms_files = lms_files[idx * batch_size:(idx + 1) * batch_size]
        batch_2dPtImgs = [get_lms_umd_image(lmsImg_train, bb, is_grayscale=0) for bb in batLms_files]
        batch_2dPtImgs = np.array(batch_2dPtImgs).astype(np.float32)
        pts2d_target = [get_lms_umd_crop_pts(bb) for bb in batLms_files]
        pts2d_target = _numpy_to_cuda(np.array(pts2d_target).astype(np.float32))
        lmsMap = [obtain_map2(idx, pts2d_target[idx], batch_2dPtImgs[idx]) for idx in range(len(pts2d_target))]
        comInput2 = comb_inputs2(batch_2dPtImgs, lmsMap, permu=False)
        comInput2 = comInput2.permute(0, 3, 1, 2)
        pts2d_target = _numpy_to_cuda(np.array(pts2d_target).astype(np.float32))

        target.requires_grad = False
        pts2d_target.requires_grad = False
        target = target.cuda(non_blocking=True)
        pts2d_target_ = pts2d_target.cuda(non_blocking=True)
        output = model(comInput1)
        pts2d_output = model(comInput2)

        data_time.update(time.time() - end)

        tri_loss_wpdc = 10*criterion(output, target)
        criterion_mse = nn.MSELoss(size_average=size_average).cuda()

        ####################################### added by xgtu ##########################################
        p_real, p_fake, sh_real, sh_fake, exp_real, exp_fake, sh_exp_real, sh_exp_fake = \
            get_real_fake_params(output, target)

        GenrVertex2lms3d = GenrVertex2lms().cuda()
        vertex, lms3d = GenrVertex2lms3d(pts2d_output)
        lms2d_68 = lms3d[:,:2,:]
        lms2d_18 = map_2d_18pts(lms3d, _18_indx_3d22d)
        lmsMap = [obtain_map2(idx, lms2d_68[idx], idx) for idx in range(len(lms2d_68))]
        comInput2_ = comb_inputs2(batch_2dPtImgs, lmsMap, permu=False)
        comInput2_ = comInput2_.permute(0, 3, 1, 2)
        pts2d_outputx = model(comInput2_)

        lmsWeighs = lmsMask_18pts(lms2d_18)
        projLoss = criterion_mse(lms2d_18.mul(lmsWeighs), map_2d_18pts(pts2d_target_, _18_indx_3d22d).mul(lmsWeighs))/18

        vertex_, lms3d_ = GenrVertex2lms3d(pts2d_outputx)
        lms2d_18_ = map_2d_18pts(lms3d_, _18_indx_3d22d)
        projLoss_crl = criterion_mse(lms2d_18_.mul(lmsWeighs), map_2d_18pts(pts2d_target_, _18_indx_3d22d).mul(lmsWeighs))/18
#        depths_img = get_depths_image(input[0], vertex[0], tri - 1)

        '''   
        aa = input[0].permute(1,2,0)
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(aa)
#        lms = pts2d_target[:, :, _18_indx_2d22d][0]
        lms = lms2d.data.cpu().numpy()[0]
        lms = lmss[0]
        for ind in range(18):
            ax.plot(lms[0, ind], lms[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                    markeredgecolor='black', alpha=0.8)
        ax.axis('off')

        plt.savefig(('./imgs/lms_18pts/plot' + lms_list[0].split(',')[0]))

        alpha = 0.8
        markersize = 4
        lw = 1.5
        color = 'w'
        markeredgecolor = 'black'
        img = input.data.cpu().numpy()[0]
        img = img.transpose(1,2,0)
        fig = plt.figure(figsize=plt.figaspect(.5))
        # fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)
        lms = output.data.cpu().numpy()[0]
        lms3d = GenrVertex2lms3d(target)
        lms = lms3d.data.cpu().numpy()[0]
        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            ax.plot(lms[0, l:r], lms[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            ax.plot(lms[0, l:r], lms[1, l:r], marker='o', linestyle='None', markersize=markersize, color=color,
                    markeredgecolor=markeredgecolor, alpha=alpha)

        ax.axis('off')
        '''

#        g_fake_loss_sh = adversarial_loss(discriminator_sh(sh_fake), real)
        g_fake_loss_exp = adversarial_loss(discriminator_exp(exp_fake), real)
        g_fake_loss = g_fake_loss_exp 
        loss = 0.1*(10*tri_loss_wpdc + 0.5*g_fake_loss + 0.05*(projLoss+projLoss_crl))
        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Measure discriminator's ability to classify real from generated samples
#        real_loss_sh = adversarial_loss(discriminator_sh(sh_real), real)
#        fake_loss_sh = adversarial_loss(discriminator_sh(sh_fake.detach()), fake)
        real_loss_exp = adversarial_loss(discriminator_exp(exp_real), real)
        fake_loss_exp = adversarial_loss(discriminator_exp(exp_fake.detach()), fake)
        d_loss = (real_loss_exp + fake_loss_exp) / 2
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        if i % epochs == 0:
            print('[Step:%d|Epoch:%d],lr:%.6f, loss:%.4f, tri_loss_wpdc:%.4f, g_fake_loss:%.4f, d_loss:%.4f, proj_loss:%.4f, projcrl_loss:%.4f' %
                  (i, epoch, lr, loss.data.cpu().numpy(), tri_loss_wpdc.data.cpu().numpy(), g_fake_loss.data.cpu().numpy(),
                   d_loss.data.cpu().numpy(), projLoss.data.cpu().numpy(), projLoss_crl.data.cpu().numpy()))
            print('[Step:%d|Epoch:%d],lr:%.6f, loss:%.4f, tri_loss_wpdc:%.4f, g_fake_loss:%.4f, d_loss:%.4f, proj_loss:%.4f, projcrl_loss:%.4f' %
                  (i, epoch, lr, loss.data.cpu().numpy(), tri_loss_wpdc.data.cpu().numpy(), g_fake_loss.data.cpu().numpy(),
                   d_loss.data.cpu().numpy(), projLoss.data.cpu().numpy(),  projLoss_crl.data.cpu().numpy()),
                    file=open(log_file + '2_wpdc_adverse_1local_exp_projLoss_18pts_mask_shufTrn.txt', 'a'))

        idx = idx + 1
        if(idx ==  batch_idxs):
            idx = 0


def validate(val_loader, model, criterion, epoch):
    model.eval()

    end = time.time()
    with torch.no_grad():
        losses = []
        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input)

            loss = criterion(output, target)
            losses.append(loss)

        elapse = time.time() - end
        loss = np.mean(losses)
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     f'Time {elapse:.3f}')


def main():
    # step1: define the model structure
    model = resnet50(pretrained=False, num_classes=num_classes)
    model_dict = model.state_dict()
    load = torch.load('./models/resnet50-19c8e357.pth')
    pretrained_dict = {k: v for k, v in load.items() if k not in ['conv1.bias', 'conv1.weight','fc.bias', 'fc.weight']}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    torch.cuda.set_device(devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`
    model = nn.DataParallel(model, device_ids=devices_id).cuda()  # -> GPU


    '''   
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(devices_id[0])
    model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=devices_id).cuda()
    model.load_state_dict(checkpoint)
    '''
    
    # step2: optimization: loss and optimization method
    # criterion = nn.MSELoss(size_average=size_average).cuda()
    if loss.lower() == 'wpdc':
        print(opt_style)
        criterion = WPDCLoss(opt_style=opt_style).cuda()
        logging.info('Use WPDC Loss')
    elif loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=opt_style).cuda()
        logging.info('Use VDC Loss')
    elif loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=size_average).cuda()
        logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=True)


    # step 2.1 resume
    if resume:
        if Path(resume).is_file():
            logging.info(f'=> loading checkpoint {resume}')

            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(resume)['state_dict']
            model.load_state_dict(checkpoint)

        else:
            logging.info(f'=> no checkpoint found at {resume}')

    # step3: data
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(
        root=root,
        filelists=filelists_train,
        param_fp=param_fp_train,
        lms_fp=lms_vfp_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADataset(
        root=root,
        filelists=filelists_val,
        param_fp=param_fp_val,
        lms_fp=lms_vfp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,
                            shuffle=False, pin_memory=True)

    # step4: run
    cudnn.benchmark = True
    if test_initial:
        logging.info('Testing from initial')
#        validate(val_loader, model, criterion, start_epoch)

    for epoch in range(start_epoch, epochs + 1):
        # adjust learning rate
#        adjust_learning_rate(optimizer, epoch, milestones)
        adjust_lr_exp(optimizer, base_lr, epoch, epochs, 2)

        # train for one epoch
#        validate(val_loader, model, criterion, epoch)
        train(train_loader, model, criterion, optimizer, epoch, loss)
        filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            },
            filename
        )

#        validate(val_loader, model, criterion, epoch)


if __name__ == '__main__':
    main()
