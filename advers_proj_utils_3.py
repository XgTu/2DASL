import os
import numpy as np
import math
import scipy.misc


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor
from params import *
_to_tensor = _numpy_to_cuda  # gpu

"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 32, bn=False),
            *discriminator_block(32, 32),
        )

        # The height and width of downsampled image
        self.adv_layer = nn.Sequential( nn.Linear(32*62, 1),
                                        nn.Sigmoid())

    def forward(self, params):
        params = params.view(params.shape[0], 1, params.shape[1], 1)
        out = self.model(params)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
"""
class Lms_3D22D(nn.Module):
    def __init__(self, input):
        super(Lms_3D22D, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 68*2),
        )

    def forward(self, lms_3d):
        lms_3d = lms_3d.view(lms_3d.shape[0], -1)
        lms_2d = self.model(lms_3d)
#        out = out.view(out.shape[0], -1)
        lms_2d = lms_2d.reshape(lms_2d.shape[0], 2, -1)

        return lms_2d

class Lms_3D22D_3layer(nn.Module):
    def __init__(self, input):
        super(Lms_3D22D_3layer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 68*2)
        )

    def forward(self, lms_3d):
        lms_3d = lms_3d.view(lms_3d.shape[0], -1)
        lms_2d = self.model(lms_3d)
#        out = out.view(out.shape[0], -1)
        lms_2d = lms_2d.reshape(lms_2d.shape[0], 2, -1)

        return lms_2d

class Lms_3D22D_1layer(nn.Module):
    def __init__(self, input):
        super(Lms_3D22D_1layer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 68*2),
        )

    def forward(self, lms_3d):
        lms_3d = lms_3d.view(lms_3d.shape[0], -1)
        lms_2d = self.model(lms_3d)
#        out = out.view(out.shape[0], -1)
        lms_2d = lms_2d.reshape(lms_2d.shape[0], 2, -1)

        return lms_2d


class Discriminator(nn.Module):
    def __init__(self, input):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

        # The height and width of downsampled image
        self.adv_layer = nn.Sequential(nn.Linear(1, 1),
                                        nn.Sigmoid())

    def forward(self, params):
        params = params.view(params.shape[0], -1)
        out = self.model(params)
#        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:52].view(N, -1, 1)
    alpha_exp = param[:, 52:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp


class GenrVertex2lms(nn.Module):
    std_size = 120
    def __init__(self):
        super(GenrVertex2lms, self).__init__()

        self.u = _to_tensor(u)
        self.param_mean = _to_tensor(param_mean)
        self.param_std = _to_tensor(param_std)
        self.w_shp = _to_tensor(w_shp)
        self.w_exp = _to_tensor(w_exp)

        self.keypoints = _to_tensor(keypoints)
        self.u_base = self.u[self.keypoints]
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

        self.w_shp_length = self.w_shp.shape[0] // 3

    def reconstruct_and_parse(self, input):
        # reconstruct
        param = input * self.param_std + self.param_mean
        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        return (p, offset, alpha_shp, alpha_exp)

    def forward_resample(self, input):
        (p, offset, alpha_shp, alpha_exp) = self.reconstruct_and_parse(input)

        N = input.shape[0]
        vertex_lms = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset
        vertex_lms[:, 1, :] = std_size + 1 - vertex_lms[:, 1, :]

        vertex_all = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset
        vertex_all[:, 1, :] = std_size + 1 - vertex_all[:, 1, :]
        return vertex_all, vertex_lms

    def forward(self, input):
        return self.forward_resample(input)

def convert_to_ori(lms, i):
    std_size = 120
    sx, sy, ex, ey = roi_boxs[i]
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    lms[0, :] = lms[0, :] * scale_x + sx
    lms[1, :] = lms[1, :] * scale_y + sy
    return lms


def transform(image):
    return (np.array(image)-127.5)/128.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def get_lms_umd_image(path, anno, is_grayscale = False):
    aa = anno.split(',')
    return transform(imread((path + aa[0]), is_grayscale))

def get_lms_umd_nonCrop_pts(anno):
    aa = anno.split(',')
    xx = aa[11:74]
    pts = np.empty([2, 21])
    pts[0,:] = [xx[i] for i in range(len(xx)) if (i + 1) % 3 == 1]
    pts[1,:] = [xx[i] for i in range(len(xx)) if (i + 1) % 3 == 2]
    return pts

def get_lms_celebA_image(path, anno, is_grayscale = False):
    aa = anno.split(',')
    return transform(imread((path + aa[0]), is_grayscale))

def get_lms_umd_crop_pts(anno):
    xx = anno.split(',')[1:1+68*2]
    pts = np.empty([2, 68])
    pts[0,:] = [xx[i] for i in range(len(xx)) if i % 2 == 0]
    pts[1,:] = [xx[i] for i in range(len(xx)) if i % 2 == 1]
    return pts

def get_lms_celebA_pts(anno):
    xx = anno.split(',')[1:1+10]
    pts = np.empty([2, 5])
    pts[0,:] = [xx[i] for i in range(len(xx)) if i % 2 == 0]
    pts[1,:] = [xx[i] for i in range(len(xx)) if i % 2 == 1]
    return pts

def map_2d_18pts(lms3d, _18_indx_3d22d):
    lms2d = lms3d[:,:,_18_indx_3d22d][:,:2,:]
    lms2d[:,:,7] = (lms3d[:,:2,37] + lms3d[:,:2,40])/2
    lms2d[:,:,10] = (lms3d[:,:2,43] + lms3d[:,:2,46])/2
    lms2d[:,:,16] = (lms3d[:,:2,62] + lms3d[:,:2,66])/2
    return lms2d

def map_2d_5pts(lms3d, _5_indx_3d22d):
    lms2d = lms3d[:,:,_5_indx_3d22d][:,:2,:]
    lms2d[:,:,0] = (lms3d[:,:2,37] + lms3d[:,:2,40])/2
    lms2d[:,:,1] = (lms3d[:,:2,43] + lms3d[:,:2,46])/2
    return lms2d

def get_real_fake_params(output, target):
    pose_real = target[:, :12].reshape(-1, 3, 4)
    pose_fake = output[:, :12].reshape(-1, 3, 4)
    sh_real = target[:, 12:52].reshape(-1, 40, 1)
    sh_fake = output[:, 12:52].reshape(-1, 40, 1)
    exp_real = target[:, 52:].reshape(-1, 10, 1)
    exp_fake = output[:, 52:].reshape(-1, 10, 1)  
 
    sh_exp_real = target[:, 12:].reshape(-1, 50, 1)
    sh_exp_fake = output[:, 12:].reshape(-1, 50, 1)

    return pose_real, pose_fake, sh_real, sh_fake, exp_real, exp_fake, sh_exp_real, sh_exp_fake

def lmsMask_18pts(lms2d):
    h_pts = [13, 16] # 2 points
    m_pts = [7, 10, 15, 17] # 4 points
    l_pts = [0,1,2,3,4,5,6,8,9,11,12,14] # 12 points
    mm = torch.FloatTensor(lms2d.shape)
    mm[:,:,h_pts] = 2.
    mm[:,:,m_pts] = 1.
    mm[:,:,l_pts] = 0.5

    return mm.cuda()
        


