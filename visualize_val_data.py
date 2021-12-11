import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from scipy.io import loadmat

import utils

if __name__ == "__main__":

    # Define the dataset
    baseroot_val = '/mnt/lustre/zhaoyuzhi/dataset'
    val_input_path = os.path.join(baseroot_val, 'ValidationNoisyBlocksSrgb.mat')
    val_input_images = loadmat(val_input_path)
    val_input_images = val_input_images['ValidationNoisyBlocksSrgb']
    val_output_path = os.path.join(baseroot_val, 'ValidationGtBlocksSrgb.mat')
    val_output_images = loadmat(val_output_path)
    val_output_images = val_output_images['ValidationGtBlocksSrgb']
    num_of_images = val_input_images.shape[0] * val_input_images.shape[1]
    print('The overall number of validation images:', num_of_images)

    # Pre-processing
    utils.check_path(os.path.join('val_results', 'input'))
    utils.check_path(os.path.join('val_results', 'gt'))

    # Read and save images
    for i in range(val_input_images.shape[0]):
        for j in range(val_input_images.shape[1]):
            # Read images
            true_input = val_input_images[i, j, :, :, :]        # [256, 256, 3], RGB
            true_target = val_output_images[i, j, :, :, :]      # [256, 256, 3], RGB
            true_input = cv2.cvtColor(true_input, cv2.COLOR_BGR2RGB)
            true_target = cv2.cvtColor(true_target, cv2.COLOR_BGR2RGB)
            # Save images
            save_img_path = '%d_%d.png' % (i, j)
            save_full_path = os.path.join('val_results', 'input', save_img_path)
            cv2.imwrite(save_full_path, true_input)
            save_full_path = os.path.join('val_results', 'gt', save_img_path)
            cv2.imwrite(save_full_path, true_target)
