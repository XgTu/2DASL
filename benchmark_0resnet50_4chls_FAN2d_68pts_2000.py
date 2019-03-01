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

from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw
from resnet_xgtu_4chls import resnet50

from ddfa_utils2 import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor

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


def obtain_68pts_map(pts):
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
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
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
    ff = open('./test.data/AFLW2000-3D_crop/list_landmarks_detect_testData_2000_68pts.txt')
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

            img = inputs.data.cpu().numpy()[0]
            img = img.transpose(1,2,0)
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 2, 1)
#            ax.imshow(img)
            '''
            lms = pts68[0]
            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                ax.plot(lms[0, l:r], lms[1, l:r], color='w', lw=1.5, alpha=0.7)
                ax.plot(lms[0, l:r], lms[1, l:r], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
            ax.axis('off')
            '''
            lms = pts68[0]
            for ind in range(18):
                ax.plot(lms[0, ind], lms[1, ind], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
            ax.axis('off')

            lmsMap = [obtain_68pts_map(aa) for aa in pts68]
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


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [0]

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW2000-3D_crop',
            filelists='test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=1)

        benchmark_aflw2000_params(params)

    aflw2000()


def main():
    preMol = './training_debug/2_wpdc_adverse_1local_exp_projLoss_18pts_mask_shufTrn_all_preTrn_4chls/_checkpoint_epoch_4.pth.tar'
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='mobilenet_v2', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default=preMol, type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()
