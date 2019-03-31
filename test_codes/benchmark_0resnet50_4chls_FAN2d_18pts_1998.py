#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time, os
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='1'

#from benchmark_aflw1998 import calc_nme as calc_nme_alfw1998
#from benchmark_aflw1998 import ana as ana_alfw1998
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw
from resnet_xgtu_4chls import resnet50

from ddfa_utils2 import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor
import os.path as osp
import numpy as np
from math import sqrt
from io_utils import _load

d = 'test.configs'
fail_detect = [1082-1, 1799-1]
yaws_list = _load(osp.join(d, 'AFLW2000-3D.pose.npy'))
yaws_list = [yaws_list[idx] for idx in range(len(yaws_list)) if idx not in fail_detect]
# pts21 = _load(osp.join(d, 'AFLW2000-3D.pts21.npy'))

# origin
pts68_all_ori = _load(osp.join(d, 'AFLW2000-3D.pts68.npy'))
pts68_all_ori = [pts68_all_ori[idx] for idx in range(len(pts68_all_ori)) if idx not in fail_detect]

# reannonated
pts68_all_re = _load(osp.join(d, 'AFLW2000-3D-Reannotated.pts68.npy'))
pts68_all_re = [pts68_all_re[idx] for idx in range(len(pts68_all_re)) if idx not in fail_detect]
roi_boxs = _load(osp.join(d, 'AFLW2000-3D_crop.roi_box.npy'))
roi_boxs = [roi_boxs[idx] for idx in range(len(roi_boxs)) if idx not in fail_detect]


def ana_alfw1998(nme_list):
    yaw_list_abs = np.abs(yaws_list)
    ind_yaw_1 = yaw_list_abs <= 30
    ind_yaw_2 = np.bitwise_and(yaw_list_abs > 30, yaw_list_abs <= 60)
    ind_yaw_3 = yaw_list_abs > 60

    nme_1 = nme_list[ind_yaw_1]
    nme_2 = nme_list[ind_yaw_2]
    nme_3 = nme_list[ind_yaw_3]

    mean_nme_1 = np.mean(nme_1) * 100
    mean_nme_2 = np.mean(nme_2) * 100
    mean_nme_3 = np.mean(nme_3) * 100
    # mean_nme_all = np.mean(nme_list) * 100

    std_nme_1 = np.std(nme_1) * 100
    std_nme_2 = np.std(nme_2) * 100
    std_nme_3 = np.std(nme_3) * 100
    # std_nme_all = np.std(nme_list) * 100

    mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
    mean = np.mean(mean_all)
    std = np.std(mean_all)

    s1 = '[ 0, 30]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
    s2 = '[30, 60]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
    s3 = '[60, 90]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
    # s4 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_all, std_nme_all)
    s5 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: \x1b[31m{:.3f}\x1b[0m'.format(mean, std)

    s = '\n'.join([s1, s2, s3, s5])
    print(s)

    return mean_nme_1, mean_nme_2, mean_nme_3, mean, std


def convert_to_ori(lms, i):
    std_size = 120
    sx, sy, ex, ey = roi_boxs[i]
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    lms[0, :] = lms[0, :] * scale_x + sx
    lms[1, :] = lms[1, :] * scale_y + sy
    return lms


def calc_nme_alfw1998(pts68_fit_all, option='ori'):
    if option == 'ori':
        pts68_all = pts68_all_ori
    elif option == 're':
        pts68_all = pts68_all_re
    std_size = 120

    nme_list = []

    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        sx, sy, ex, ey = roi_boxs[i]
        scale_x = (ex - sx) / std_size
        scale_y = (ey - sy) / std_size
        pts68_fit[0, :] = pts68_fit[0, :] * scale_x + sx
        pts68_fit[1, :] = pts68_fit[1, :] * scale_y + sy

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        #
        dis = pts68_fit - pts68_gt[:2, :]
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list


def get_lms_crop_pts(anno):
    xx = anno.split(',')[1:1+68*2]
    pts = np.empty([2, 68])
    pts[0,:] = [xx[i] for i in range(len(xx)) if i % 2 == 0]
    pts[1,:] = [xx[i] for i in range(len(xx)) if i % 2 == 1]
    return pts

def map_2d_18pts(lms68, _18_indx_3d22d):
    lms18 = lms68[:,:,_18_indx_3d22d][:,:2,:]
    lms18[:,:,7] = (lms68[:,:2,37] + lms68[:,:2,40])/2
    lms18[:,:,10] = (lms68[:,:2,43] + lms68[:,:2,46])/2
    lms18[:,:,16] = (lms68[:,:2,62] + lms68[:,:2,66])/2
    return lms18

def map_2d_18pts_2d(lms2d_68):
    _18_indx_3d22d = [17, 19, 21, 22, 24, 26, 36, 40, 39, 42, 46, 45, 31, 30, 35, 48, 66, 54]
    lms2d = lms2d_68[:,_18_indx_3d22d]
    lms2d[:,7] = (lms2d_68[:,37] + lms2d_68[:,40])/2
    lms2d[:,10] = (lms2d_68[:,43] + lms2d_68[:,46])/2
    lms2d[:,16] = (lms2d_68[:,62] + lms2d_68[:,66])/2
    return lms2d


def obtain_18pts_map(pts):
    pts = map_2d_18pts_2d(pts)
    ptsMap = np.zeros([120, 120]) - 1
    indx = np.int32(np.floor(pts))
#    print(pts)
    ptsMap[indx[1], indx[0]] = 1

    '''
    aa = ptsMap
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(aa)

    for ind in range(18):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
    ax.axis('off')

    cv2.imwrite(('./imgs/lms_18pts/' + lms.split(',')[0]), ptsMap*255)
    '''
    return ptsMap

def comb_inputs(imgs, lmsMaps, permu=False):
    lmsMaps = np.array(lmsMaps).astype(np.float32)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [np.dstack((imgs[idx].cpu().numpy(),lmsMaps[idx])) for idx in range(imgs.shape[0])]
    outputs = np.array(outputs).astype(np.float32)
    return outputs


def extract_param(checkpoint_fp, root='', filelists=None, arch='resnet50', num_classes=62, device_ids=[0],
                  batch_size=1, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['res_state_dict']
    torch.cuda.set_device(device_ids[0])
    model = resnet50(pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    ff = open('./test.data/detect_testData_1998.txt')
#    ff = open('./test.data/AFLW2000-3D_crop/list_landmarks_align_AFLW2000_3D_crop_pts_xgtu_21.txt')
    lmsList = ff.readlines()
    ff.close()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
#            print(idx)
            batLms_files = lmsList[idx * batch_size:(idx + 1) * batch_size]
            pts68 = [get_lms_crop_pts(bb) for bb in batLms_files]
            pts68 = np.array(pts68).astype(np.float32)
            pts68 = pts68[:,:2,:]
            pts68[pts68>119] = 119
#            print(pts68)
            inputs = inputs.cuda()
            '''
            img = inputs.data.cpu().numpy()[0]
            img = img.transpose(1,2,0)
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(img)

            lms = pts68[0]
            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                ax.plot(lms[0, l:r], lms[1, l:r], color='w', lw=1.5, alpha=0.7)
                ax.plot(lms[0, l:r], lms[1, l:r], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
            ax.axis('off')

            lms = pts68[0]
            for ind in range(18):
                ax.plot(lms[0, ind], lms[1, ind], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
            ax.axis('off')
            '''
            lmsMap = [obtain_18pts_map(aa) for aa in pts68]
            comInput1 = comb_inputs(inputs, lmsMap, permu=True)
            comInput1 = _numpy_to_cuda(comInput1)
            comInput1 = comInput1.permute(0, 3, 1, 2)

            output = model(comInput1)
            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)
        

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw1998(outputs):
    return ana_alfw1998(calc_nme_alfw1998(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw1998_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw1998(outputs)


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [0]

    def aflw1998():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW2000-3D_crop',
            filelists='test.data/AFLW1998-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=1)

        benchmark_aflw1998_params(params)

    aflw1998()


def main():
    preMol = '../models/2DASL_checkpoint_epoch_allParams_stage2.pth.tar'
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='mobilenet_v2', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default=preMol, type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()
