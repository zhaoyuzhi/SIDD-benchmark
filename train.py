import argparse
import yaml
from easydict import EasyDict as edict
import os

import trainer

def attatch_to_config(opt, yaml_args):
    # Pre-train, saving, and loading parameters
    opt.network = yaml_args.name
    opt.save_path = yaml_args.Training.save_path
    opt.sample_path = yaml_args.Training.sample_path
    opt.save_mode = yaml_args.Training.save_mode
    opt.save_by_epoch = yaml_args.Training.save_by_epoch
    opt.save_by_iter = yaml_args.Training.save_by_iter
    opt.load_name = ""
    # Training parameters
    opt.multi_gpu = yaml_args.Training.multi_gpu
    opt.cudnn_benchmark = yaml_args.Training.cudnn_benchmark
    opt.epochs = yaml_args.Training.epochs
    opt.train_batch_size = yaml_args.Training.train_batch_size
    opt.val_batch_size = yaml_args.Training.val_batch_size
    opt.lr_g = yaml_args.Training.lr_g
    opt.lr_d = yaml_args.Training.lr_d
    opt.b1 = yaml_args.Training.b1
    opt.b2 = yaml_args.Training.b2
    opt.weight_decay = yaml_args.Training.weight_decay
    opt.lr_decrease_epoch = yaml_args.Training.lr_decrease_epoch
    opt.num_workers = yaml_args.Training.num_workers
    opt.lambda_multiloss = yaml_args.Training.lambda_multiloss
    # Dataset parameters
    opt.baseroot_train = yaml_args.Dataset.baseroot_train
    opt.baseroot_val = yaml_args.Dataset.baseroot_val
    opt.patch_per_image = yaml_args.Dataset.patch_per_image
    opt.cutblur_prob = yaml_args.Dataset.cutblur_prob
    opt.cutblur_size = yaml_args.Dataset.cutblur_size
    
if __name__ == "__main__":
    
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, \
        default = './options/SFM_SGN_v3.yaml', \
            help = 'yaml_path')
    parser.add_argument('--network', type = str, default = 'DeblurGAN', help = 'network name')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 150, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_multiloss', type = float, default = 0.5, help = 'coefficient for GAN Loss')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot_train', type = str, \
        default = 'F:\dataset, task related\Denoising Dataset\SIDD\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data', \
            help = 'input image baseroot')
    parser.add_argument('--baseroot_val', type = str, \
        default = 'E:\\submitted papers\\QuadBayer Deblur\\data\\val', \
            help = 'output image baseroot')
    parser.add_argument('--patch_per_image', type = int, default = 4, \
        help = 'the number of patches extracted from the same image')
    parser.add_argument('--crop_size', type = int, default = 320, help = 'crop_size')
    parser.add_argument('--cutblur_prob', type = float, default = 0.5, help = 'cutblur_prob')
    parser.add_argument('--cutblur_size', type = int, default = 120, help = 'cutblur_size')
    opt = parser.parse_args()
    
    with open(opt.yaml_path, mode = 'r') as f:
        yaml_args = edict(yaml.load(f))

    attatch_to_config(opt, yaml_args)
    print(opt)

    trainer.L1_Trainer(opt)
    