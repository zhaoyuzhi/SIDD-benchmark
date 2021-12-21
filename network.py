import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

import network_redi_module as redi_module
import network_sgnlarge_module as sgnlarge_module

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ----------------------------------------
#              Basic Network
# ----------------------------------------
class SFM_SGN_res_blocks(nn.Module):
    def __init__(self, nf=8):
        super(SFM_SGN_res_blocks, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf),
                                    nn.Conv2d(nf, nf, 3, padding=1, stride=1))
        self.block2 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf),
                                    nn.Conv2d(nf, nf, 3, padding=1, stride=1))
        self.block3 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf),
                                    nn.Conv2d(nf, nf, 3, padding=1, stride=1))
        self.block4 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf),
                                    nn.Conv2d(nf, nf, 3, padding=1, stride=1))

    def forward(self, x):
        xr = self.block1(x)
        x = x + xr
        xr = self.block2(x)
        x = x + xr
        xr = self.block3(x)
        x = x + xr
        xr = self.block4(x)
        x = x + xr
        return x

class SFM_SGN_v3(nn.Module):
    def __init__(self, nf = 8):
        print("model name: %s" % (self.__class__.__name__))
        super(SFM_SGN_v3, self).__init__()
        in_channel = 3
        out_channel = 3

        self.downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.blockh_1 = nn.Sequential(nn.Conv2d(in_channel, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.blockh_2 = nn.Sequential(nn.Conv2d(in_channel, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.blockh_3 = nn.Sequential(nn.Conv2d(in_channel, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.block1 = SFM_SGN_res_blocks(nf=nf)
        self.block2 = SFM_SGN_res_blocks(nf=nf)
        self.block3 = SFM_SGN_res_blocks(nf=nf)

        self.convt = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.Conv2d(nf, nf, 3, padding=1, stride=1),
                                   nn.Conv2d(nf, out_channel, 3, padding=1, stride=1))

        self.print_network()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    def forward(self, x):
        xd2 = self.downsampleby2(x)
        xd4 = self.downsampleby2(xd2)

        x3 = self.block3(self.blockh_3(xd4))
        x3 = self.upsampleby2(x3)

        x2 = self.block2(self.blockh_2(xd2) + x3)
        x2 = self.upsampleby2(x2)

        x0 = self.blockh_1(x)
        x1 = self.block1(x0 + x2)
        x1 = x1 + x0
        x1 = self.convt(x1)
        y = x1
        return y, y


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class SFM_SGN_v3_twiceDWT_fixch(nn.Module):
    def __init__(self, nf=8):
        super(SFM_SGN_v3_twiceDWT_fixch, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        
        in_channel = 3
        out_channel = 3

        self.downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dwt = DWT()
        self.idwt = IWT()
        
        self.blockh_1 = nn.Sequential(nn.Conv2d(in_channel, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.blockh_2 = nn.Sequential(nn.Conv2d(in_channel * 4, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.blockh_3 = nn.Sequential(nn.Conv2d(in_channel * 16, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.block1 = SFM_SGN_res_blocks(nf=nf)
        self.block2 = SFM_SGN_res_blocks(nf=nf)
        self.block3 = SFM_SGN_res_blocks(nf=nf)

        self.block_recover_2 = nn.Sequential(nn.Conv2d(nf // 4, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.block_recover_3 = nn.Sequential(nn.Conv2d(nf // 4, nf, 3, padding=1, stride=1), nn.PReLU(num_parameters=nf))

        self.convt = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, stride=1), nn.Conv2d(nf, nf, 3, padding=1, stride=1),
                                   nn.Conv2d(nf, out_channel, 3, padding=1, stride=1))

        self.print_network()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    def forward(self, x):
        xd2 = self.dwt(x)
        xd4 = self.dwt(xd2)

        x3 = self.block3(self.blockh_3(xd4))
        x3 = self.idwt(x3)
        x3 = self.block_recover_3(x3)

        x2 = self.block2(self.blockh_2(xd2) + x3)
        x2 = self.idwt(x2)
        x2 = self.block_recover_2(x2)

        x0 = self.blockh_1(x)
        x1 = self.block1(x0 + x2)
        x1 = x1 + x0
        x1 = self.convt(x1)
        y = x1
        return y, y


# ----------------------------------------
#             Advanced Network
# ----------------------------------------
class SFM_SGN_v3_ch8_lapacian_4level_unshared(nn.Module):
    def __init__(self):
        super(SFM_SGN_v3_ch8_lapacian_4level_unshared, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        self.downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = SFM_SGN_v3(nf=8)
        self.modeld2 = SFM_SGN_v3(nf=8)
        self.modeld4 = SFM_SGN_v3(nf=8)
        self.modeld8 = SFM_SGN_v3(nf=8)

    def forward(self, x):
        xd2 = self.downsampleby2(x)
        xd4 = self.downsampleby2(xd2)
        xd8 = self.downsampleby2(xd4)

        outd8, _ = self.modeld8(xd8)
        outd8_up = self.upsampleby2(outd8)
        xd4 = xd4 + outd8_up

        outd4, _ = self.modeld4(xd4)
        outd4_up = self.upsampleby2(outd4)
        xd2 = xd2 + outd4_up

        outd2, _ = self.modeld2(xd2)
        outd2_up = self.upsampleby2(outd2)
        x0 = x + outd2_up

        out, _ = self.model(x0)
        final_out = x0 + out
        return final_out, [x0, xd2, xd4]


class SFM_SGN_v3_twiceDWT_fixch_ch8_lapacian_4level_unshared(nn.Module):
    def __init__(self):
        super(SFM_SGN_v3_twiceDWT_fixch_ch8_lapacian_4level_unshared, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        self.downsampleby2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = SFM_SGN_v3_twiceDWT_fixch(nf=8)
        self.modeld2 = SFM_SGN_v3_twiceDWT_fixch(nf=8)
        self.modeld4 = SFM_SGN_v3_twiceDWT_fixch(nf=8)
        self.modeld8 = SFM_SGN_v3_twiceDWT_fixch(nf=8)

    def forward(self, x):
        xd2 = self.downsampleby2(x)
        xd4 = self.downsampleby2(xd2)
        xd8 = self.downsampleby2(xd4)

        outd8, _ = self.modeld8(xd8)
        outd8_up = self.upsampleby2(outd8)
        xd4 = xd4 + outd8_up

        outd4, _ = self.modeld4(xd4)
        outd4_up = self.upsampleby2(outd4)
        xd2 = xd2 + outd4_up

        outd2, _ = self.modeld2(xd2)
        outd2_up = self.upsampleby2(outd2)
        x0 = x + outd2_up

        out, _ = self.model(x0)
        final_out = x0 + out
        return final_out, [x0, xd2, xd4]


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class SFM_SGN_v3_ch8_lapacian_4level_unshared_blurpool(nn.Module):
    def __init__(self):
        super(SFM_SGN_v3_ch8_lapacian_4level_unshared_blurpool, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        self.antialiased_downsampleby2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1),
            BlurPool(channels=3, stride=2))
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = SFM_SGN_v3(nf=8)
        self.modeld2 = SFM_SGN_v3(nf=8)
        self.modeld4 = SFM_SGN_v3(nf=8)
        self.modeld8 = SFM_SGN_v3(nf=8)

    def forward(self, x):
        xd2 = self.antialiased_downsampleby2(x)
        xd4 = self.antialiased_downsampleby2(xd2)
        xd8 = self.antialiased_downsampleby2(xd4)

        outd8, _ = self.modeld8(xd8)
        outd8_up = self.upsampleby2(outd8)
        xd4 = xd4 + outd8_up

        outd4, _ = self.modeld4(xd4)
        outd4_up = self.upsampleby2(outd4)
        xd2 = xd2 + outd4_up

        outd2, _ = self.modeld2(xd2)
        outd2_up = self.upsampleby2(outd2)
        x0 = x + outd2_up

        out, _ = self.model(x0)
        final_out = x0 + out
        return final_out, [x0, xd2, xd4]


class SFM_SGN_v3_ch8_lapacian_4level_unshared_FC_blurpool(nn.Module):
    def __init__(self):
        super(SFM_SGN_v3_ch8_lapacian_4level_unshared_FC_blurpool, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        self.antialiased_downsampleby2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1),
            BlurPool(channels=3, stride=2))
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = SFM_SGN_v3(nf=8)
        self.modeld2 = SFM_SGN_v3(nf=8)
        self.modeld4 = SFM_SGN_v3(nf=8)
        self.modeld8 = SFM_SGN_v3(nf=8)

    def forward(self, x):
        xd2 = self.antialiased_downsampleby2(x)
        xd4 = self.antialiased_downsampleby2(xd2)
        xd8 = self.antialiased_downsampleby2(xd4)

        outd8, _ = self.modeld8(xd8)
        outd8_up2 = self.upsampleby2(outd8)
        outd8_up4 = self.upsampleby2(outd8_up2)
        outd8_up8 = self.upsampleby2(outd8_up4)
        xd4 = xd4 + outd8_up2

        outd4, _ = self.modeld4(xd4)
        outd4_up2 = self.upsampleby2(outd4)
        outd4_up4 = self.upsampleby2(outd4_up2)
        xd2 = xd2 + outd4_up2 + outd8_up4

        outd2, _ = self.modeld2(xd2)
        outd2_up2 = self.upsampleby2(outd2)
        x0 = x + outd2_up2 + outd4_up4 + outd8_up8

        out, _ = self.model(x0)
        final_out = x0 + out
        return final_out, [x0, xd2, xd4]
        

class SFM_SGN_v3_ch8_lapacian2_4level_unshared_blurpool(nn.Module):
    def __init__(self):
        super(SFM_SGN_v3_ch8_lapacian2_4level_unshared_blurpool, self).__init__()
        print("model name: %s" % (self.__class__.__name__))
        self.antialiased_downsampleby2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1),
            BlurPool(channels=3, stride=2))
        self.antialiased_upsampleby2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BlurPool(channels=3, stride=1))
        self.model = SFM_SGN_v3(nf=8)
        self.modeld2 = SFM_SGN_v3(nf=8)
        self.modeld4 = SFM_SGN_v3(nf=8)
        self.modeld8 = SFM_SGN_v3(nf=8)

    def get_laplacian(self, x):
        # downsample
        xd2 = self.antialiased_downsampleby2(x)
        xd4 = self.antialiased_downsampleby2(xd2)
        xd8 = self.antialiased_downsampleby2(xd4)
        # upsample
        xd8_up4 = self.antialiased_upsampleby2(xd8)
        lap4 = xd4 - xd8_up4
        xd4_up2 = self.antialiased_upsampleby2(xd8_up4)
        lap2 = xd2 - xd4_up2
        xd2_up = self.antialiased_upsampleby2(xd4_up2)
        lap = x - xd2_up
        return lap, lap2, lap4, xd8

    # how to reconstruct:
    # first:
    # xd8 -> neural network (4-th level) -> f(xd8) -> antialiased_upsampleby2 -> f(xd8)_up4
    # input for upper stage: xd4_new = f(xd8)_up4 + lap4
    # second:
    # xd4_new -> neural network (3-rd level) -> f(xd4_new) -> antialiased_upsampleby2 -> f(xd4_new)_up2
    # input for upper stage: xd2_new = f(xd4_new)_up2 + lap2
    # third:
    # xd2_new -> neural network (2-nd level) -> f(xd2_new) -> antialiased_upsampleby2 -> f(xd2_new)_up
    # input for upper stage: x_new = f(xd2_new)_up + lap
    # final:
    # x_new -> neural network (1-st level) -> output
    def forward(self, x):
        lap, lap2, lap4, xd8 = self.get_laplacian(x)

        outd8, _ = self.modeld8(xd8)
        xd8_processed = xd8 + outd8
        outd8_up4 = self.antialiased_upsampleby2(xd8_processed)
        xd4_new = outd8_up4 + lap4

        outd4, _ = self.modeld4(xd4_new)
        xd4_processed = xd4_new + outd4
        outd4_up2 = self.antialiased_upsampleby2(xd4_processed)
        xd2_new = outd4_up2 + lap2

        outd2, _ = self.modeld2(xd2_new)
        xd2_processed = xd2_new + outd2
        outd2_up = self.antialiased_upsampleby2(xd2_processed)
        x_new = outd2_up + lap

        out, _ = self.model(x_new)
        final_out = x_new + out
        return final_out, [outd2, outd4, outd8]


# ----------------------------------------
#              Other Network
# ----------------------------------------
class REDI(nn.Module):
    def __init__(self):
        super(REDI, self).__init__()
        self.up4 = nn.PixelShuffle(4)
        self.up2 = nn.PixelShuffle(2)
        
        self.conv32x = nn.Sequential(        
            redi_module.conv_layer(3072, 128, kernel_size=3, groups=128, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False),
            redi_module.conv_layer(128, 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False)
            )
                        
        self.RDB1 = redi_module.RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = redi_module.RDB(nChannels=64, nDenselayer=5, growthRate=32)
        self.RDB3 = redi_module.RDB(nChannels=64, nDenselayer=5, growthRate=32)

        self.rdball = redi_module.conv_layer(int(64*3), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False)
        
        self.conv_rdb8x = redi_module.conv_layer(int(64//16), 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False)
        
        self.resblock8x = redi_module.ResBlock(192)
        
        self.conv32_8_cat = nn.Sequential(
            redi_module.conv_layer(256, 32, kernel_size=3, groups=4, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False),
            redi_module.conv_layer(32, 192, kernel_size=3, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False),
            self.up4)                      
        
        self.conv2x = redi_module.conv_layer(12, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False)
        
        self.conv_2_8_32 = redi_module.conv_layer(24, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False)
    
        self.print_network()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    def downshuffle(self,var,r):
        b,c,h,w = var.size()
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w).contiguous()
        
    def forward(self, low):
        
        residual = low
        low2x = self.downshuffle(low,2)
        
        # 32x branch starts
        low32x_beforeRDB = self.conv32x(self.downshuffle(low2x,16))
        rdb1 = self.RDB1(low32x_beforeRDB)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb8x = torch.cat((rdb1,rdb2,rdb3),dim=1)
        rdb8x = self.rdball(rdb8x)+low32x_beforeRDB
        rdb8x = self.up4(rdb8x)
        rdb8x = self.conv_rdb8x(rdb8x)
        
        # 8x branch starts
        low8x = self.resblock8x(self.downshuffle(low2x,4))
        cat_32_8 = torch.cat((low8x,rdb8x),dim=1).contiguous()
        
        b,c,h,w = cat_32_8.size()
        G=2
        cat_32_8 = cat_32_8.view(b, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        cat_32_8 = self.conv32_8_cat(cat_32_8)
        
        # 2x branch starts
        low2x = torch.cat((self.conv2x(low2x), cat_32_8),dim=1)
        low2x = self.up2(self.conv_2_8_32(low2x))

        output = residual - low2x
        
        return output, output


class SGNLarge(nn.Module):
    def __init__(self):
        super(SGNLarge, self).__init__()
        in_channels = 3
        start_channels = 32
        out_channels = 3
        # Top subnetwork, K = 3
        self.top1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        self.top2 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        self.top3 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        # Middle subnetwork, K = 2
        self.mid1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        self.mid2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 2 + 2 ** 3 / 4)), start_channels * (2 ** 2), 3, 1, 1)
        self.mid3 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        self.mid4 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        # Bottom subnetwork, K = 1
        self.bot1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        self.bot2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 1 + 2 ** 2 / 4)), start_channels * (2 ** 1), 3, 1, 1)
        self.bot3 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        self.bot4 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        # Mainstream
        self.main1 = sgnlarge_module.Conv2dLayer(in_channels, start_channels, 3, 1, 1)
        self.main2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 0 + 2 ** 1 / 4)), start_channels, 3, 1, 1)
        self.main3 = nn.ModuleList([sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1)])
        self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        for i in range(2):                            # add m conv blocks
            self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        self.main4 = sgnlarge_module.Conv2dLayer(start_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = sgnlarge_module.pixel_unshuffle(x, 2)              # out: batch * 12 * 128 * 128
        x2 = sgnlarge_module.pixel_unshuffle(x, 4)              # out: batch * 48 * 64 * 64
        x3 = sgnlarge_module.pixel_unshuffle(x, 8)              # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top2(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid3(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot3(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        for model in self.main3:
            x = model(x)                                        # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        out = residual - x
        return out, out


class SGNSmall(nn.Module):
    def __init__(self):
        super(SGNSmall, self).__init__()
        in_channels = 3
        start_channels = 8
        out_channels = 3
        # Top subnetwork, K = 3
        self.top1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        self.top2 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        self.top3 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 3, 1, 1)
        # Middle subnetwork, K = 2
        self.mid1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        self.mid2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 2 + 2 ** 3 / 4)), start_channels * (2 ** 2), 3, 1, 1)
        self.mid3 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        self.mid4 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1)
        # Bottom subnetwork, K = 1
        self.bot1 = sgnlarge_module.Conv2dLayer(in_channels * (4 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        self.bot2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 1 + 2 ** 2 / 4)), start_channels * (2 ** 1), 3, 1, 1)
        self.bot3 = sgnlarge_module.ResConv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        self.bot4 = sgnlarge_module.Conv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1)
        # Mainstream
        self.main1 = sgnlarge_module.Conv2dLayer(in_channels, start_channels, 3, 1, 1)
        self.main2 = sgnlarge_module.Conv2dLayer(int(start_channels * (2 ** 0 + 2 ** 1 / 4)), start_channels, 3, 1, 1)
        self.main3 = nn.ModuleList([sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1)])
        self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        for i in range(2):                            # add m conv blocks
            self.main3.append(sgnlarge_module.Conv2dLayer(start_channels, start_channels, 3, 1, 1))
        self.main4 = sgnlarge_module.Conv2dLayer(start_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = sgnlarge_module.pixel_unshuffle(x, 2)              # out: batch * 12 * 128 * 128
        x2 = sgnlarge_module.pixel_unshuffle(x, 4)              # out: batch * 48 * 64 * 64
        x3 = sgnlarge_module.pixel_unshuffle(x, 8)              # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top2(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid3(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot3(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        for model in self.main3:
            x = model(x)                                        # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        out = residual - x
        return out, out


if __name__ == "__main__":

    #net = SFM_SGN_v3_ch8_lapacian_4level_unshared().cuda()
    #net = SFM_SGN_v3_twiceDWT_fixch_ch8_lapacian_4level_unshared().cuda()
    net = SGNLarge().cuda()
    #net = BlurPool(channels=3, stride=1).cuda()

    x = torch.randn(1, 3, 320, 320).cuda()
    y = net(x)
    if isinstance(y, tuple):
        y = y[0]
    print(y.shape)
    