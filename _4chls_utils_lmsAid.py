import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor



def obtain_map(lms):
    lms = lms.strip()
    xx = lms.split(',')[1:]
    pts = np.empty([2, 68])
    pts[0,:] = [xx[i] for i in range(len(xx)) if (i) % 2 == 0]
    pts[1,:] = [xx[i] for i in range(len(xx)) if (i) % 2 == 1]

    ptsMap = np.zeros([120, 120])-1
    indx = np.int32(np.floor(pts))
    indx[indx>119] = 119
    indx[indx<1] = 1
    ptsMap[indx[1], indx[0]] = 1

    '''
    img = cv2.imread('./train_aug_120x120/' + lms.split(',')[0])
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    aa = ptsMap
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(aa)
    
    for ind in range(68):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                markeredgecolor='black', alpha=0.8)
    ax.axis('off')

    plt.savefig(('./imgs/lmsMaps/' + lms.split(',')[0]))
    '''

    return ptsMap

def obtain_map2(idx, pts, img):
    ptsMap = torch.zeros([120,120]) - 1
    indx = np.int32(pts.floor().int())
    indx[indx>119] = 119
    indx[indx<1] = 1
    ptsMap[indx[1], indx[0]] = 1
    '''
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    aa = ptsMap
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(aa)

    for ind in range(68):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                markeredgecolor='black', alpha=0.8)
    ax.axis('off')

#    cv2.imwrite(('./imgs/bat3lmsMaps/' + 'xx.jpg'), ptsMap*255)
    plt.savefig(('./imgs/bat3lmsMaps/' + str(idx) + '.jpg'))
    '''
    return ptsMap

def comb_inputs(imgs, lmsMaps, permu=False):
    lmsMaps = np.array(lmsMaps).astype(np.float32)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [np.dstack((imgs[idx],lmsMaps[idx])) for idx in range(imgs.shape[0])]
    outputs = np.array(outputs).astype(np.float32)
    return outputs

def comb_inputs2(imgs, lmsMaps, permu=False):
    imgs = _numpy_to_cuda(imgs)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [torch.cat((imgs[idx], lmsMaps[idx].unsqueeze(2).cuda()), 2) for idx in range(imgs.shape[0])]
    return torch.stack(outputs)

def obtain_18pts_map2(pts, smp = True):
    _18_indx_2d22d = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,17,18,19]
    if(smp == True):
        pts = pts[:, _18_indx_2d22d]
    else:
        pts = pts
#    ptsMap = np.zeros([120, 120])-1
#    indx = np.int32(np.floor(pts))
    ptsMap = torch.zeros([120,120]) - 1
    indx = np.int32(pts.floor().int())
    indx[indx>119] = 119
    indx[indx<1] = 1
    ptsMap[indx[1], indx[0]] = 1
    

    aa = ptsMap
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(aa)
 
    for ind in range(18):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                markeredgecolor='black', alpha=0.8)
    ax.axis('off')

    cv2.imwrite(('./imgs/lms_18pts/' + 'xx.jpg'), ptsMap*255)

    return ptsMap
