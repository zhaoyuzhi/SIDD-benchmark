import argparse
import yaml
from easydict import EasyDict as edict
import os

import trainer
import utils

def get_model_name(opt, yaml_args):
    tail_name = opt.yaml_path.split('/')[-1].split('.')[0]
    load_folder = os.path.join(yaml_args.load_name, tail_name)
    load_name_list = utils.get_files(load_folder)
    for i in range(len(load_name_list)):
        if 'epoch100' in load_name_list[i]:
            return load_name_list[i]
        elif 'epoch50' in load_name_list[i]:
            return load_name_list[i]
        else:
            print('no valid model found')

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.load_name = get_model_name(opt, yaml_args)
    print('Searched model name:', opt.load_name)
    # Validation parameters
    opt.whether_save = yaml_args.Validation.whether_save
    opt.saveroot = yaml_args.Validation.saveroot
    opt.val_batch_size = 1
    opt.num_workers = 0
    opt.enable_patch = yaml_args.Validation.enable_patch
    opt.patch_size = yaml_args.Validation.patch_size
    # Dataset parameters
    opt.baseroot_val = yaml_args.Dataset.baseroot_val

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, \
        default = './options/SGN.yaml', \
            help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'SGN', help = 'network name')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # Validation parameters
    parser.add_argument('--whether_save', type = bool, default = True, help = 'whether saving generated images')
    parser.add_argument('--saveroot', type = str, default = './val_results', help = 'saving path that is a folder')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'val_batch_size, fixed to 1')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'num_workers, fixed to 0')
    parser.add_argument('--enable_patch', type = bool, default = True, help = 'whether use patch for validation')
    parser.add_argument('--patch_size', type = int, default = 512, help = 'patch_size')
    # Dataset parameters
    parser.add_argument('--baseroot_val', type = str, \
        default = 'F:\\dataset, task related\\Denoising Dataset\\SIDD', \
            help = 'output image baseroot')
    opt = parser.parse_args()

    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))

    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.Valer(opt)
    #trainer.Tester(opt)
    