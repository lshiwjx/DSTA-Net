import math
import sys

# sys.path.extend(['../'])
# from method_choose.model_choose import rm_module
import torch
import torch.nn as nn


def BN3d(channel, sy=False):
    # 3x3x3 convolution with padding
    if sy:
        return SynchronizedBatchNorm3d(channel)
    else:
        return nn.BatchNorm3d(channel)


# >50 using bottleblock
class BottleBlock(nn.Module):
    def __init__(self, channel_in, channel_mid, channel_out, downsample=None, group=1, stride=1, deform=False,
                 dgroup=1):
        super(BottleBlock, self).__init__()

        if deform == False:
            self.conv1 = nn.Conv3d(channel_in, channel_mid, kernel_size=1, padding=0, bias=False)
            self.conv2 = nn.Conv3d(channel_mid, channel_mid, kernel_size=3, groups=group, padding=1, bias=False,
                                   stride=stride)
            self.conv3 = nn.Conv3d(channel_mid, channel_out, kernel_size=1, padding=0, bias=False)

        else:
            self.conv_off1 = nn.Conv3d(channel_in, dgroup * 3 * 1, kernel_size=3, stride=1, padding=1)
            self.conv1 = ConvOffset3d(channel_in, channel_mid, kernel_size=1, groups=1, padding=0, bias=False,
                                      channel_per_group=channel_in // dgroup)

            self.conv_off2 = nn.Conv3d(channel_mid, dgroup * 3 * 27, kernel_size=3, stride=stride,
                                       padding=1, bias=True, dilation=1)
            self.conv2 = ConvOffset3d(channel_mid, channel_mid, kernel_size=3, groups=group, padding=1, bias=False,
                                      stride=stride, dilation=1, channel_per_group=channel_mid // dgroup)

            self.conv_off3 = nn.Conv3d(channel_mid, dgroup * 3 * 1, kernel_size=3, stride=1, padding=1)
            self.conv3 = ConvOffset3d(channel_mid, channel_out, kernel_size=1, padding=0, bias=False,
                                      channel_per_group=channel_mid // dgroup)
        self.bn1 = BN3d(channel_mid)
        self.bn2 = BN3d(channel_mid)
        self.bn3 = BN3d(channel_out)
        self.relu = nn.ReLU(inplace=True)
        # self.down = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.downsample = downsample
        self.ratio = channel_out // channel_in
        self.deform = deform

        if self.deform:
            # nn.init.constant_(self.conv_off0.weight.data, 0)
            # nn.init.constant_(self.conv_off0.bias.data, 0)
            nn.init.constant_(self.conv_off1.weight.data, 1e-6)
            nn.init.constant_(self.conv_off1.bias.data, 1e-6)
            nn.init.constant_(self.conv_off2.weight.data, 1e-6)
            nn.init.constant_(self.conv_off2.bias.data, 1e-6)
            nn.init.constant_(self.conv_off3.weight.data, 1e-6)
            nn.init.constant_(self.conv_off3.bias.data, 1e-6)
        nn.init.kaiming_normal_(self.conv1.weight.data, mode='fan_out')
        nn.init.kaiming_normal_(self.conv2.weight.data, mode='fan_out')
        nn.init.kaiming_normal_(self.conv3.weight.data, mode='fan_out')

    def forward(self, x):
        # residual = torch.cat([x for _ in range(self.ratio)], 1)
        residual = x

        if self.deform:
            off = self.conv_off1(x)
            out = self.conv1(x, off)
            out = self.bn1(out)
            out = self.relu(out)

            off = self.conv_off2(out)
            out = self.conv2(out, off)
            out = self.bn2(out)
            out = self.relu(out)

            off = self.conv_off3(out)
            out = self.conv3(out, off)
            out = self.bn3(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        out = self.relu(out)

        if self.deform:
            return out, off
        else:
            return out, None


class ResNetXt1013d(nn.Module):
    def __init__(self, class_num):
        super(ResNetXt1013d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = BN3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        downsample0 = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            BN3d(256)
        )
        self.layer10 = BottleBlock(64, 128, 256, downsample0, 32)
        self.layer11 = BottleBlock(256, 128, 256, None, 32)
        self.layer12 = BottleBlock(256, 128, 256, None, 32)

        downsample1 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=False),
            BN3d(512)
        )
        self.layer20 = BottleBlock(256, 256, 512, downsample1, 32, stride=2)
        self.layer21 = BottleBlock(512, 256, 512, None, 32)
        self.layer22 = BottleBlock(512, 256, 512, None, 32)
        self.layer23 = BottleBlock(512, 256, 512, None, 32)

        downsample2 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=1, stride=2, bias=False),
            BN3d(1024)
        )
        self.layer30 = BottleBlock(512, 512, 1024, downsample2, 32, stride=2)
        self.layer31 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer32 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer33 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer34 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer35 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer36 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer37 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer38 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer39 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer310 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer311 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer312 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer313 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer314 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer315 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer316 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer317 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer318 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer319 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer320 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer321 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer322 = BottleBlock(1024, 512, 1024, None, 32)

        downsample3 = nn.Sequential(
            nn.Conv3d(1024, 2048, kernel_size=1, stride=2, bias=False),
            BN3d(2048)
        )
        self.layer40 = BottleBlock(1024, 1024, 2048, downsample3, 32, stride=2)
        self.layer41 = BottleBlock(2048, 1024, 2048, None, 32)
        self.layer42 = BottleBlock(2048, 1024, 2048, None, 32)

        self.dropout = nn.Dropout(inplace=True, p=0.9)
        self.fc = nn.Linear(2048, class_num)
        self.layers = []

        # init weight
        nn.init.kaiming_normal_(self.conv1.weight.data, mode='fan_out')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer10(x)
        x, _ = self.layer11(x)
        x, _ = self.layer12(x)
        x, _ = self.layer20(x)
        x, y1 = self.layer21(x)
        x, y2 = self.layer22(x)
        x, y3 = self.layer23(x)

        x, _ = self.layer30(x)
        x, _ = self.layer31(x)
        x, _ = self.layer32(x)
        x, _ = self.layer33(x)
        x, _ = self.layer34(x)
        x, _ = self.layer35(x)
        x, _ = self.layer36(x)
        x, _ = self.layer37(x)
        x, _ = self.layer38(x)
        x, _ = self.layer39(x)
        x, _ = self.layer310(x)
        x, _ = self.layer311(x)
        x, _ = self.layer312(x)
        x, _ = self.layer313(x)
        x, _ = self.layer314(x)
        x, _ = self.layer315(x)
        x, _ = self.layer316(x)
        x, _ = self.layer317(x)
        x, _ = self.layer318(x)
        x, _ = self.layer319(x)

        x, y1 = self.layer320(x)
        x, y2 = self.layer321(x)
        x, y3 = self.layer322(x)

        x, y1 = self.layer40(x)
        x, y2 = self.layer41(x)
        x, y3 = self.layer42(x)

        x = x.view(x.size(0), 2048, -1)
        x = torch.mean(x, 2)
        x = self.dropout(x)

        x = self.fc(x)

        return x  # 1461M

if __name__ == '__main__':

    import os
    from torch.autograd import Variable

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model = ResNetXt1013d(249)
    x = Variable(torch.zeros(1, 3, 32, 112, 112))
    r = model(x)
    print(r.shape)
