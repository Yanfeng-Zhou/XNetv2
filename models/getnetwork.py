import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'XNetv2' or network == 'xnetv2':
        net = xnetv2(in_channels, num_classes)
    # 3d networks
    elif network == 'XNetv2_3D_min' or network == 'xnetv2_3d_min':
        net = xnetv2_3d_min(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
