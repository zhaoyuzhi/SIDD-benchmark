import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def get_image_list(path):
    noisy_imglist = []
    gt_imglist = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fullpath = os.path.join(root, filespath)
            #fullpath_exclude_tail = fullpath.split('.PNG')[0]
            if 'NOISY' in fullpath:
                noisy_imglist.append(fullpath)
                gt_fullpath = fullpath.replace('NOISY', 'GT')
                gt_imglist.append(gt_fullpath)
    return noisy_imglist, gt_imglist

def get_image_list_val(path):
    noisy_imglist = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fullpath = os.path.join(root, filespath)
            #fullpath_exclude_tail = fullpath.split('.PNG')[0]
            if 'NOISY_SRGB' in fullpath:
                noisy_imglist.append(fullpath)
    return noisy_imglist

class SIDD_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt

        # Build training dataset
        self.noisy_imglist, self.gt_imglist = get_image_list(opt.baseroot_train)
        print('training imglist length:', len(self.noisy_imglist))

    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def __getitem__(self, index):
        
        # Rand index
        rid = random.randint(0, len(self.noisy_imglist) - 1)

        # Path of images
        noisy_img_path = self.noisy_imglist[rid]
        gt_img_path = self.gt_imglist[rid]

        # Read images
        noisy_img = cv2.imread(noisy_img_path)
        gt_img = cv2.imread(gt_img_path)

        # Build data lists
        input_list = []
        gt_list = []
        h, w = noisy_img.shape[:2]
        for id1 in range(self.opt.patch_per_image):
            # Extract patches
            rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 2)
            noisy_patch = noisy_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
            gt_patch = gt_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
            # Cutblur
            if self.opt.cutblur_prob > 0:
                p = random.random()
                if p < self.opt.cutblur_prob:
                    try:
                        rand_h2, rand_w2 = self.random_crop_start(self.opt.crop_size, self.opt.crop_size, self.opt.cutblur_size, 2)
                        noisy_patch[rand_h2:rand_h+self.opt.cutblur_size, rand_w2:rand_w+self.opt.cutblur_size, :] = \
                            gt_patch[rand_h2:rand_h+self.opt.cutblur_size, rand_w2:rand_w+self.opt.cutblur_size, :]
                    except:
                        continue
            # Normalization
            noisy_patch = noisy_patch / 255.0
            gt_patch = gt_patch / 255.0
            # To tensor
            noisy_patch = torch.from_numpy(noisy_patch).float().permute(2, 0, 1).contiguous()
            gt_patch = torch.from_numpy(gt_patch).float().permute(2, 0, 1).contiguous()
            # Add to lists
            input_list.append(noisy_patch)
            gt_list.append(gt_patch)

        # Concatenate
        for id2 in range(len(input_list)):
            if id2 == 0:
                input_batch = input_list[id2].unsqueeze(0)
                gt_batch = gt_list[id2].unsqueeze(0)
            else:
                input_batch = torch.cat((input_batch, input_list[id2].unsqueeze(0)), 0)
                gt_batch = torch.cat((gt_batch, gt_list[id2].unsqueeze(0)), 0)
            input_batch = input_batch.contiguous()
            gt_batch = gt_batch.contiguous()

        return input_batch, gt_batch
    
    def __len__(self):
        return 50000

class SIDD_ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt

        # Build training dataset
        self.noisy_imglist = get_image_list_val(opt.baseroot_val)
        print('training imglist length:', len(self.noisy_imglist))

    def __getitem__(self, index):
        
        # Path of images
        noisy_img_path = self.noisy_imglist[index]
        save_img_path = noisy_img_path.split('/')[-1]

        # Read images
        noisy_img = cv2.imread(noisy_img_path)

        # Normalization
        noisy_img = noisy_img / 255.0

        # To tensor
        noisy_img = torch.from_numpy(noisy_img).float().permute(2, 0, 1).contiguous()

        return noisy_img, save_img_path
    
    def __len__(self):
        return len(self.noisy_imglist)

if __name__ == "__main__":

    path = '0001_NOISY_SRGB_010.PNG'
    path2 = path.replace('NOISY', 'GT')
    print(path2)

    fullpath = '/0001_001_S6_00100_00060_3200_L/0001_NOISY_SRGB_010.PNG'
    fullpath2 = fullpath.split('.PNG')[0]
    print(fullpath2)
