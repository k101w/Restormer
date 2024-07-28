import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
import random

src = '/data/interns/kewenwu/Restormer/Denoising/Datasets/train/denosing_dataset/input_crops'
src2 = '/data/interns/kewenwu/Restormer/Denoising/Datasets/train/denosing_dataset/target_crops'
tar1 = '/data/interns/kewenwu/Restormer/Denoising/Datasets/val/denosing_dataset/input_crops'
tar2 = '/data/interns/kewenwu/Restormer/Denoising/Datasets/val/denosing_dataset/target_crops'
os.makedirs(tar1, exist_ok=True)
os.makedirs(tar2, exist_ok=True)

files = natsorted(glob(os.path.join(src, '*.png')))
validation_images = random.sample(files, 200)
for file_ in validation_images:
    img = cv2.imread(file_)
    os.remove(file_)
    filename = os.path.split(file_)[-1]
    savename = os.path.join(tar1,filename)
    cv2.imwrite(savename, img)
    img2 = cv2.imread(os.path.join(src2,filename))
    os.remove(os.path.join(src2,filename))
    savename2 = os.path.join(tar2,filename)
    cv2.imwrite(savename2, img2)
