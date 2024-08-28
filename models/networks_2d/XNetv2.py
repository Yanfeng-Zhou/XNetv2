import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class XNetv2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(XNetv2, self).__init__()

        # main network
        self.M_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.M_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.M_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.M_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.M_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.M_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.M_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.M_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.M_Up4 = up_conv(ch_in=512, ch_out=256)
        self.M_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.M_Up3 = up_conv(ch_in=256, ch_out=128)
        self.M_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.M_Up2 = up_conv(ch_in=128, ch_out=64)
        self.M_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.M_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # L network
        self.L_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.L_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.L_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.L_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.L_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.L_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.L_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.L_Up4 = up_conv(ch_in=512, ch_out=256)
        self.L_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.L_Up3 = up_conv(ch_in=256, ch_out=128)
        self.L_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.L_Up2 = up_conv(ch_in=128, ch_out=64)
        self.L_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.L_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # H network
        self.H_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.H_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.H_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.H_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.H_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.H_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.H_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.H_Up4 = up_conv(ch_in=512, ch_out=256)
        self.H_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.H_Up3 = up_conv(ch_in=256, ch_out=128)
        self.H_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.H_Up2 = up_conv(ch_in=128, ch_out=64)
        self.H_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.H_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # fusion
        self.M_H_Conv1 = conv_block(ch_in=128, ch_out=64)
        self.M_H_Conv2 = conv_block(ch_in=256, ch_out=128)
        self.M_L_Conv3 = conv_block(ch_in=512, ch_out=256)
        self.M_L_Conv4 = conv_block(ch_in=1024, ch_out=512)

    def forward(self, x_main, x_L, x_H):
        # main encoder
        M_x1 = self.M_Conv1(x_main)
        M_x2 = self.M_Maxpool(M_x1)
        M_x2 = self.M_Conv2(M_x2)
        M_x3 = self.M_Maxpool(M_x2)
        M_x3 = self.M_Conv3(M_x3)
        M_x4 = self.M_Maxpool(M_x3)
        M_x4 = self.M_Conv4(M_x4)
        M_x5 = self.M_Maxpool(M_x4)
        M_x5 = self.M_Conv5(M_x5)

        # L encoder
        L_x1 = self.L_Conv1(x_L)
        L_x2 = self.L_Maxpool(L_x1)
        L_x2 = self.L_Conv2(L_x2)
        L_x3 = self.L_Maxpool(L_x2)
        L_x3 = self.L_Conv3(L_x3)
        L_x4 = self.L_Maxpool(L_x3)
        L_x4 = self.L_Conv4(L_x4)
        L_x5 = self.L_Maxpool(L_x4)
        L_x5 = self.L_Conv5(L_x5)

        # H encoder
        H_x1 = self.H_Conv1(x_H)
        H_x2 = self.H_Maxpool(H_x1)
        H_x2 = self.H_Conv2(H_x2)
        H_x3 = self.H_Maxpool(H_x2)
        H_x3 = self.H_Conv3(H_x3)
        H_x4 = self.H_Maxpool(H_x3)
        H_x4 = self.H_Conv4(H_x4)
        H_x5 = self.H_Maxpool(H_x4)
        H_x5 = self.H_Conv5(H_x5)

        # fusion
        M_H_x1 = torch.cat((M_x1, H_x1), dim=1)
        M_H_x1 = self.M_H_Conv1(M_H_x1)
        M_H_x2 = torch.cat((M_x2, H_x2), dim=1)
        M_H_x2 = self.M_H_Conv2(M_H_x2)
        M_L_x3 = torch.cat((M_x3, L_x3), dim=1)
        M_L_x3 = self.M_L_Conv3(M_L_x3)
        M_L_x4 = torch.cat((M_x4, L_x4), dim=1)
        M_L_x4 = self.M_L_Conv4(M_L_x4)

        # main decoder

        M_d5 = self.M_Up5(M_x5)
        M_d5 = torch.cat((M_L_x4, M_d5), dim=1)
        M_d5 = self.M_Up_conv5(M_d5)

        M_d4 = self.M_Up4(M_d5)
        M_d4 = torch.cat((M_L_x3, M_d4), dim=1)
        M_d4 = self.M_Up_conv4(M_d4)

        M_d3 = self.M_Up3(M_d4)
        M_d3 = torch.cat((M_H_x2, M_d3), dim=1)
        M_d3 = self.M_Up_conv3(M_d3)

        M_d2 = self.M_Up2(M_d3)
        M_d2 = torch.cat((M_H_x1, M_d2), dim=1)
        M_d2 = self.M_Up_conv2(M_d2)

        M_d1 = self.M_Conv_1x1(M_d2)

        # L decoder
        L_d5 = self.L_Up5(L_x5)
        L_d5 = torch.cat((M_L_x4, L_d5), dim=1)
        L_d5 = self.L_Up_conv5(L_d5)

        L_d4 = self.L_Up4(L_d5)
        L_d4 = torch.cat((M_L_x3, L_d4), dim=1)
        L_d4 = self.L_Up_conv4(L_d4)

        L_d3 = self.L_Up3(L_d4)
        L_d3 = torch.cat((L_x2, L_d3), dim=1)
        L_d3 = self.L_Up_conv3(L_d3)

        L_d2 = self.L_Up2(L_d3)
        L_d2 = torch.cat((L_x1, L_d2), dim=1)
        L_d2 = self.L_Up_conv2(L_d2)

        L_d1 = self.L_Conv_1x1(L_d2)

        # H decoder
        H_d5 = self.H_Up5(H_x5)
        H_d5 = torch.cat((H_x4, H_d5), dim=1)
        H_d5 = self.H_Up_conv5(H_d5)

        H_d4 = self.H_Up4(H_d5)
        H_d4 = torch.cat((H_x3, H_d4), dim=1)
        H_d4 = self.H_Up_conv4(H_d4)

        H_d3 = self.H_Up3(H_d4)
        H_d3 = torch.cat((M_H_x2, H_d3), dim=1)
        H_d3 = self.H_Up_conv3(H_d3)

        H_d2 = self.H_Up2(H_d3)
        H_d2 = torch.cat((M_H_x1, H_d2), dim=1)
        H_d2 = self.H_Up_conv2(H_d2)

        H_d1 = self.H_Conv_1x1(H_d2)

        return M_d1, L_d1, H_d1

def xnetv2(in_channels, num_classes):
    model = XNetv2(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


# if __name__ == '__main__':
#     model = xnetv2(1,10)
#     model.eval()
#     input1 = torch.rand(2,1,128,128)
#     input2 = torch.rand(2,1,128,128)
#     input3 = torch.rand(2,1,128,128)
#     output1, output2, output3 = model(input1, input2, input3)
#     output1 = output1.data.cpu().numpy()
#     # print(output)
#     print(output1.shape)
