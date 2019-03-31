#!/usr/bin/env python3
# coding: utf-8

from benchmark_0resnet50_4chls_FAN2d_18pts_1998 import extract_param
from ddfa_utils import reconstruct_vertex
from io_utils import _dump, _load
import os.path as osp
from skimage import io
import matplotlib.pyplot as plt
from benchmark_aflw2000 import convert_to_ori
from mpl_toolkits.mplot3d import Axes3D
import cv2
import scipy.io as sio


def aflw2000():
    arch = 'resnet50'
    device_ids = [0]
    checkpoint_fp = '../models/2DASL_checkpoint_epoch_allParams_stage2.pth.tar'

    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='test.data/AFLW2000-3D_crop',
        filelists='test.data/AFLW1998-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        batch_size=1)
    _dump('results/params_aflw2000_xgtu.npy', params)


def draw_landmarks():
    root_ori = 'test.data/AFLW-2000-3D'
    root = 'test.data/AFLW2000-3D_crop'
    filelists = 'test.data/AFLW2000-3D_crop.list'
    fns = open(filelists).read().strip().split('\n')
    params = _load('results/params_aflw2000_xgtu.npy')

    for i in range(len(fns)):
        plt.close()
        img_fp = osp.join(root_ori, fns[i])
        img = io.imread(img_fp)
        lms = reconstruct_vertex(params[i], dense=False)
        lms = convert_to_ori(lms, i)
        # lms = convert_to_ori_frmMat(lms, [root + fns[i]])

        # print(lms.shape)
        fig = plt.figure(figsize=plt.figaspect(.5))
        # fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)

        alpha = 0.8
        markersize = 1.5
        lw = 1.2
        color = 'w'
        markeredgecolor = 'b'

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            ax.plot(lms[0, l:r], lms[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            ax.plot(lms[0, l:r], lms[1, l:r], marker='o', linestyle='None', markersize=markersize, color=color,
                    markeredgecolor=markeredgecolor, alpha=alpha)

        ax.axis('off')

        plt.savefig('results/3dLandmarks_proj_resnet50_4chls/' + img_fp.split('/')[-1])

def gen_3d_vertex():
    filelists = 'test.data/AFLW2000-3D_crop.list'
    root = 'test.data/AFLW-2000-3D/'
    fns = open(filelists).read().strip().split('\n')
    params = _load('results/params_aflw2000_xgtu.npy')

    for i in range(100):
        fn = fns[i]
        vertex = reconstruct_vertex(params[i], dense=True)
        wfp = osp.join('results/AFLW-2000-3D_vertex/', fn.replace('.jpg', '.mat'))
        print(wfp)
        sio.savemat(wfp, {'vertex': vertex})


def main():
    # step1: extract params
    aflw2000()

    # step2: draw landmarks
    draw_landmarks()

    # step3: visual 3d vertex
    gen_3d_vertex()


if __name__ == '__main__':
    main()
