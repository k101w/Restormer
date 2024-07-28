import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
import pdb

# src = "/data/interns/kewenwu/grey/DBS_Dataset"
# src = "/data/interns/kewenwu/grey/denoise_sharp"
# src = "/data/interns/kewenwu/grey/ksharp_pair"

# src = "/data/interns/kewenwu/grey/DBS_Dataset"
# src = '/data/interns/kewenwu/grey/DBS_Dataset_generated_pair'
# src = '/data/interns/kewenwu/grey/DBS_Dataset_generated_pair'
# src = '/data/interns/kewenwu/grey/HQ_data_generate_pair'
# src = '/data/interns/kewenwu/grey/HQ_data_ST_layer_generate_pair'
# src = '/data/interns/kewenwu/grey/kSharp_training_data_selected_pair'
src = '/data/interns/kewenwu/grey/paired_data_for_kewen_4x_generate_pair'
tar = 'Datasets/train/denosing_dataset'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)
if 'DBS' in src and 'generated' not in src:
    lr_files = natsorted(glob(os.path.join(src, '*',  'Data', '*.png')))
    hr_files = natsorted(glob(os.path.join(src, '*',  'Labels', '*.png')))

else:
    lr_files = natsorted(glob(os.path.join(src, 'Data', '*.png')))
    hr_files = natsorted(glob(os.path.join(src, 'Labels', '*.png')))
# if 'DBS' in src:
#     files = natsorted(glob(os.path.join(src, '*',  '*', '*.png')))
# else:
#     files = natsorted(glob(os.path.join(src, '*', '*.png')))

# lr_files, hr_files = [], []
# for file_ in files:
#     filename = os.path.split(file_)[-1]
#     if 'labels' in filename:
#     # if 'sharp_nle40_denoise_aggressive' in filename:
#         hr_files.append(file_)
#     else:
#         lr_files.append(file_)

files = [(i, j) for i, j in zip(lr_files, hr_files)]
patch_size = 512
overlap = 128
p_max = 0

def save_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    # pdb.set_trace()
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    if lr_img.shape[0] != hr_img.shape[0]:
       print('shape error')
       return
    # if filename == '0.0_0.0_FOV0.256_PD512_SR_1X_FA1_SA_ZERO_Raw0_16bit_20-11-03':
    #     cv2.imwrite('data.png',lr_img)
    #     cv2.imwrite('labels.png',hr_img)
    #     pdb.set_trace()
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int_))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int_))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                try:
                    cv2.imwrite(hr_savename, hr_patch)
                except:
                    pdb.set_trace()

    else:
        lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename = os.path.join(hr_tar, filename + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

from joblib import Parallel, delayed
import multiprocessing
num_cores = 20
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))
# for file_ in tqdm(files):
#     save_files(file_)