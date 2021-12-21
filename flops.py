import argparse
import torch
from easydict import EasyDict as edict
from thop import profile
from thop import clever_format

import network

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type = str, \
        default = 'SGNSmall', \
            help = 'net_name')
    parser.add_argument('--h', type = int, default = 512, help = 'h resolution')
    parser.add_argument('--w', type = int, default = 512, help = 'w resolution')
    opt = parser.parse_args()

    # SFM_SGN_v3 macs: 2.243G params: 16.195K
    # SFM_SGN_v3_ch8_lapacian_4level_unshared macs: 2.979G params: 64.780K
    # SFM_SGN_v3_twiceDWT_fixch_ch8_lapacian_4level_unshared macs: 3.189G params: 81.612K
    # REDI macs: 4.274G params: 1.404M
    # SGNLarge macs: 61.843G params: 3.942M
    # SGNSmall macs: 4.640G params: 341.803K

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    
    # Define the network
    generator = getattr(network, opt.network)()
    generator.cuda()

    for param in generator.parameters():
        param.requires_grad = False

    # forward propagation
    input = torch.randn(1, 3, opt.h, opt.w).cuda()

    macs, params = profile(generator, inputs = (input, ))
    macs_1, params_1 = clever_format([macs, params], "%.3f")
    print(opt.network, 'macs:', macs_1, 'params:', params_1)
