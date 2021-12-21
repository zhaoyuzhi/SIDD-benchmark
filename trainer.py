import time
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from scipy.io import loadmat

import dataset
import utils
import network

def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    tail_name = opt.network
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.SIDD_Dataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (input_batch, gt_batch) in enumerate(train_loader):

            # To device
            input_batch = input_batch.cuda()
            gt_batch = gt_batch.cuda()

            # Process patch
            if len(input_batch.shape) == 5:
                _, _, C, H, W = input_batch.shape # B, B, C, H, W
                input_batch = input_batch.view(-1, C, H, W)
                gt_batch = gt_batch.view(-1, C, H, W)

            # Train Generator
            optimizer_G.zero_grad()
            gen_batch, _ = generator(input_batch)
            
            # L1 Loss
            L1_Loss = criterion_L1(gen_batch, gt_batch)

            # Overall Loss and optimize
            loss = L1_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), L1_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [input_batch, gen_batch, gt_batch]
            name_list = ['input', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

def Trainer_Multilevel(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    tail_name = opt.yaml_path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    antialiased_downsampleby2 = nn.Sequential(
        nn.AvgPool2d(kernel_size=2, stride=1),
        network.BlurPool(channels=3, stride=2))

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        downsampleby2 = nn.DataParallel(downsampleby2)
        downsampleby2 = downsampleby2.cuda()
        antialiased_downsampleby2 = nn.DataParallel(antialiased_downsampleby2)
        antialiased_downsampleby2 = antialiased_downsampleby2.cuda()
    else:
        generator = generator.cuda()
        downsampleby2 = downsampleby2.cuda()
        antialiased_downsampleby2 = antialiased_downsampleby2.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.SIDD_Dataset(opt)

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (input_batch, gt_batch) in enumerate(train_loader):

            # To device
            input_batch = input_batch.cuda()
            gt_batch = gt_batch.cuda()

            # Process patch
            if len(input_batch.shape) == 5:
                _, _, C, H, W = input_batch.shape # B, B, C, H, W
                input_batch = input_batch.view(-1, C, H, W)
                gt_batch = gt_batch.view(-1, C, H, W)
            
            # Process patch only for multilevel loss
            if 'Multilevel' in tail_name:
                if 'blurpool' in tail_name:
                    halfsize_gt_batch = antialiased_downsampleby2(gt_batch)
                    onefourthsize_gt_batch = antialiased_downsampleby2(halfsize_gt_batch)
                else:
                    halfsize_gt_batch = downsampleby2(gt_batch)
                    onefourthsize_gt_batch = downsampleby2(halfsize_gt_batch)

            # Train Generator
            optimizer_G.zero_grad()
            gen_batch, multilevel_gen_batch = generator(input_batch)
            
            # L1 Loss
            if 'Multilevel' in tail_name:
                Multilevel_L1_Loss = criterion_L1(multilevel_gen_batch[0], gt_batch) + \
                    criterion_L1(multilevel_gen_batch[1], halfsize_gt_batch) + \
                        criterion_L1(multilevel_gen_batch[2], onefourthsize_gt_batch)
                if 'tanhl1' in tail_name:
                    L1_Loss = criterion_L1(torch.tanh(gen_batch), torch.tanh(gt_batch))
                else:
                    L1_Loss = criterion_L1(gen_batch, gt_batch)
                loss = L1_Loss + 0.5 * Multilevel_L1_Loss
            else:
                if 'tanhl1' in tail_name:
                    L1_Loss = criterion_L1(torch.tanh(gen_batch), torch.tanh(gt_batch))
                    loss = L1_Loss
                else:
                    L1_Loss = criterion_L1(gen_batch, gt_batch)
                    loss = L1_Loss

            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f] [L1/TanhL1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), L1_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [input_batch, gen_batch, gt_batch]
            name_list = ['input', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

def Trainer_Multilevel2(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    tail_name = opt.yaml_path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.save_path, tail_name)
    sample_folder = os.path.join(opt.sample_path, tail_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    antialiased_downsampleby2 = nn.Sequential(
        nn.AvgPool2d(kernel_size=2, stride=1),
        network.BlurPool(channels=3, stride=2))

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        downsampleby2 = nn.DataParallel(downsampleby2)
        downsampleby2 = downsampleby2.cuda()
        antialiased_downsampleby2 = nn.DataParallel(antialiased_downsampleby2)
        antialiased_downsampleby2 = antialiased_downsampleby2.cuda()
    else:
        generator = generator.cuda()
        downsampleby2 = downsampleby2.cuda()
        antialiased_downsampleby2 = antialiased_downsampleby2.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = '%s_epoch%d_bs%d.pth' % (opt.network, epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = '%s_iter%d_bs%d.pth' % (opt.network, iteration, opt.train_batch_size)
        save_model_path = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.SIDD_Dataset(opt)

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (input_batch, gt_batch) in enumerate(train_loader):

            # To device
            input_batch = input_batch.cuda()
            gt_batch = gt_batch.cuda()

            # Process patch
            if len(input_batch.shape) == 5:
                _, _, C, H, W = input_batch.shape # B, B, C, H, W
                input_batch = input_batch.view(-1, C, H, W)
                gt_batch = gt_batch.view(-1, C, H, W)
            
            # Process patch only for multilevel loss
            if 'Multilevel2' in tail_name:
                if 'blurpool' in tail_name:
                    halfsize_gt_batch = antialiased_downsampleby2(gt_batch)
                    onefourthsize_gt_batch = antialiased_downsampleby2(halfsize_gt_batch)
                    oneeighthsize_gt_batch = antialiased_downsampleby2(onefourthsize_gt_batch)
                else:
                    halfsize_gt_batch = downsampleby2(gt_batch)
                    onefourthsize_gt_batch = downsampleby2(halfsize_gt_batch)
                    oneeighthsize_gt_batch = downsampleby2(onefourthsize_gt_batch)

            # Train Generator
            optimizer_G.zero_grad()
            gen_batch, multilevel_gen_batch = generator(input_batch)
            
            # L1 Loss
            if 'Multilevel2' in tail_name:
                Multilevel_L1_Loss = criterion_L1(multilevel_gen_batch[0], halfsize_gt_batch) + \
                    criterion_L1(multilevel_gen_batch[1], onefourthsize_gt_batch) + \
                        criterion_L1(multilevel_gen_batch[2], oneeighthsize_gt_batch)
                if 'tanhl1' in tail_name:
                    L1_Loss = criterion_L1(torch.tanh(gen_batch), torch.tanh(gt_batch))
                else:
                    L1_Loss = criterion_L1(gen_batch, gt_batch)
                loss = L1_Loss + 0.5 * Multilevel_L1_Loss
            else:
                if 'tanhl1' in tail_name:
                    L1_Loss = criterion_L1(torch.tanh(gen_batch), torch.tanh(gt_batch))
                    loss = L1_Loss
                else:
                    L1_Loss = criterion_L1(gen_batch, gt_batch)
                    loss = L1_Loss
            
            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f] [L1/TanhL1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), L1_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [input_batch, gen_batch, gt_batch]
            name_list = ['input', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), \
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

def Valer(opt):

    # ----------------------------------------
    #                Prepararing
    # ----------------------------------------

    # configurations
    tail_name = opt.yaml_path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.saveroot, tail_name)
    if opt.whether_save:
        utils.check_path(save_folder)

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    generator = generator.cuda()

    # Define the dataset
    val_input_path = os.path.join(opt.baseroot_val, 'ValidationNoisyBlocksSrgb.mat')
    val_input_images = loadmat(val_input_path)
    val_input_images = val_input_images['ValidationNoisyBlocksSrgb']
    val_output_path = os.path.join(opt.baseroot_val, 'ValidationGtBlocksSrgb.mat')
    val_output_images = loadmat(val_output_path)
    val_output_images = val_output_images['ValidationGtBlocksSrgb']
    num_of_images = val_input_images.shape[0] * val_input_images.shape[1]
    print('The overall number of validation images:', num_of_images)

    # ----------------------------------------
    #                Validation
    # ----------------------------------------

    # forward
    val_PSNR = 0
    val_SSIM = 0
    
    # For loop validation
    for i in range(val_input_images.shape[0]):
        for j in range(val_input_images.shape[1]):
            
            # Read images
            true_input = val_input_images[i, j, :, :, :]        # [256, 256, 3], RGB
            true_target = val_output_images[i, j, :, :, :]      # [256, 256, 3], RGB
            save_img_path = '%d_%d.png' % (i, j)

            # To device
            true_input = cv2.cvtColor(true_input, cv2.COLOR_BGR2RGB)
            true_input = true_input / 255.0
            true_input = torch.from_numpy(true_input).float().permute(2, 0, 1).unsqueeze(0).contiguous()
            true_input = true_input.cuda()

            true_target = cv2.cvtColor(true_target, cv2.COLOR_BGR2RGB)
            true_target = true_target / 255.0
            true_target = torch.from_numpy(true_target).float().permute(2, 0, 1).unsqueeze(0).contiguous()
            true_target = true_target.cuda()

            # Forward
            out = generator(true_input)       
            if isinstance(out, list) or isinstance(out, tuple):
                out = out[0]

            # Save the image (BCHW -> HWC)
            if opt.whether_save:
                save_img = out[0, :, :, :].clone().data.permute(1, 2, 0).cpu().numpy()
                save_img = np.clip(save_img, 0, 1)
                save_img = (save_img * 255).astype(np.uint8)
                save_full_path = os.path.join(save_folder, save_img_path)
                cv2.imwrite(save_full_path, save_img)

            # PSNR
            # print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
            this_PSNR = utils.psnr(out, true_target, 1) * true_target.shape[0]
            val_PSNR += this_PSNR
            this_SSIM = utils.ssim(out, true_target) * true_target.shape[0]
            val_SSIM += this_SSIM
            print('The %d-th image: Name: %s PSNR: %.5f, SSIM: %.5f' % (i + 1, save_img_path, this_PSNR, this_SSIM))

    val_PSNR = val_PSNR / num_of_images
    val_SSIM = val_SSIM / num_of_images
    print('The average of %s: PSNR: %.5f, average SSIM: %.5f' % (opt.load_name, val_PSNR, val_SSIM))

def Tester(opt):

    # ----------------------------------------
    #                Prepararing
    # ----------------------------------------

    # configurations
    tail_name = opt.yaml_path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.saveroot, tail_name)
    if opt.whether_save:
        utils.check_path(save_folder)

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    generator = generator.cuda()

    # Define the dataset
    valset = dataset.SIDD_ValDataset(opt)
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                Validation
    # ----------------------------------------
    
    # For loop training
    for i, (true_input, save_img_path) in enumerate(val_loader):

        # To device
        true_input = true_input.cuda()
        save_img_path = save_img_path[0]

        # Forward
        with torch.no_grad():
            if opt.enable_patch:
                _, _, H, W = true_input.shape 
                patch_size = opt.patch_size
                assert patch_size % 4 == 0
                patchGen = utils.PatchGenerator(H, W, patch_size)
                out = torch.zeros_like(true_input)
                for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                    img_patch = true_input[:, :, h:h+patch_size, w:w+patch_size]
                    out_patch = generator(img_patch)
                    if isinstance(out_patch, list):
                        out_patch = out_patch[0]
                    out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                        out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
            else:
                out = generator(true_input)       
                if isinstance(out, list):
                    out = out[0]
        
        # Save the image (BCHW -> HWC)
        if opt.whether_save:
            save_img = out[0, :, :, :].clone().data.permute(1, 2, 0).cpu().numpy()
            save_img = np.clip(save_img, 0, 1)
            save_img = (save_img * 255).astype(np.uint8)
            save_full_path = os.path.join(save_folder, save_img_path)
            cv2.imwrite(save_full_path, save_img)
        