import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import cv2
import scipy

if __name__ == '__main__':
    image_shape = (160, 576)
    data_dir = './data'
    BATCH_SIZE = 16

    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    for imgs in get_batches_fn(BATCH_SIZE):
        img = np.concatenate(imgs[0],axis=0)/255.0
        lbl = np.concatenate(imgs[1],axis=0)
        lbl = np.dstack((lbl,lbl[:,:,0].reshape(-1,lbl.shape[1],1)))*1.0
        lbl[:,:,0]=0
        lbl[:,:,2]=0
        merged = np.concatenate((img,lbl),axis=1)*255
        cv2.imwrite('images/batch.jpg',merged)
        break
