import numpy as np
import scipy.io as sio

yaws_list = np.load('./AFLW2000-3D.pose.npy')
yaw_list_abs = np.abs(yaws_list)

sio.savemat('AFLW2000-3D.pose.mat', {'yaws': yaw_list_abs})