import numpy as np
import pickle as pickle


imgPath = './train.configs/train_aug_120x120.list.train'
paramPath = './train.configs/param_all_norm.pkl'
fail_FAN_ids = 'train.configs/FAN_fail_detect_ID_for_trainData.txt'

decon_img_path = './train.configs/train_aug_120x120_list_decon.txt'
decon_pram_path = './train.configs/param_all_norm_decon.pkl'


f1 = open(imgPath)
img_list = f1.readlines()
f1.close()

par_list = pickle.load(open(paramPath, 'rb'))

fail_dect_lists = open(fail_FAN_ids).readlines()
fail_dect_ids = [int(item) for item in fail_dect_lists]


ff = open(decon_img_path, 'w')
[ff.write(img_list[idx]) for idx in range(len(img_list)) if idx not in fail_dect_ids]
ff.close()

par_list_decon = [par_list[idx] for idx in range(len(par_list)) if idx not in fail_dect_ids]
par_list_decon = np.array(par_list_decon).astype(np.float32)
fff = open(decon_pram_path, 'wb')
pickle.dump(par_list_decon,fff)

aaa = 1