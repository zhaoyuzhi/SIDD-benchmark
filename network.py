import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import partial

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

        out, _ = self.model(x)
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

        out, _ = self.model(x)
        final_out = x0 + out
        return final_out, [x0, xd2, xd4]


if __name__ == "__main__":

    #net = SFM_SGN_v3_ch8_lapacian_4level_unshared().cuda()
    #net = SFM_SGN_v3_twiceDWT_fixch_ch8_lapacian_4level_unshared().cuda()
    net = SFM_SGN_v3_twiceDWT_fixch().cuda()

    x = torch.randn(1, 3, 320, 320).cuda()
    y = net(x)
    if isinstance(y, tuple):
        y = y[0]
    print(y.shape)
    