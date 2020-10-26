import torch.nn as nn
import torch


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight.data, mode='fan_out')
        nn.init.constant_(self.bn.weight.data, 1)
        nn.init.constant_(self.bn.bias.data, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NoreluConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(NoreluConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes)

        nn.init.kaiming_normal_(self.conv.weight.data, mode='fan_out')
        nn.init.constant_(self.bn.weight.data, 1)
        nn.init.constant_(self.bn.bias.data, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DeformConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dgroup=1):
        super(DeformConv3d, self).__init__()
        self.conv_off = nn.Conv3d(in_planes, dgroup * 3 * kernel_size * kernel_size * kernel_size,
                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.conv = ConvOffset3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=False, channel_per_group=in_planes // dgroup)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.conv_off.weight.data, 1e-6)
        nn.init.constant_(self.conv_off.bias.data, 0)

        nn.init.kaiming_normal_(self.conv.weight.data, mode='fan_out')
        nn.init.constant_(self.bn.weight.data, 1)
        nn.init.constant_(self.bn.bias.data, 0)

    def forward(self, x):
        off = self.conv_off(x)
        x = self.conv(x, off)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            BasicConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            BasicConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f_norelu(nn.Module):
    def __init__(self):
        super(Mixed_4f_norelu, self).__init__()

        self.branch0 = nn.Sequential(
            NoreluConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            NoreluConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            NoreluConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            NoreluConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class D_Mixed_5b(nn.Module):
    def __init__(self):
        super(D_Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            DeformConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            DeformConv3d(832, 160, kernel_size=1, stride=1),
            DeformConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            DeformConv3d(832, 32, kernel_size=1, stride=1),
            DeformConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            DeformConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c_norelu(nn.Module):
    def __init__(self):
        super(Mixed_5c_norelu, self).__init__()

        self.branch0 = nn.Sequential(
            NoreluConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            NoreluConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            NoreluConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            NoreluConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class D_Mixed_5c(nn.Module):
    def __init__(self):
        super(D_Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            DeformConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            DeformConv3d(832, 192, kernel_size=1, stride=1),
            DeformConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            DeformConv3d(832, 48, kernel_size=1, stride=1),
            DeformConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            DeformConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class I3D(nn.Module):
    def __init__(self, num_classes=400, out_feature=False, input_channel=3, spatial_squeeze=True):
        super(I3D, self).__init__()
        self.features = nn.Sequential(
            BasicConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3),  # (64, 32, 112, 112)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (64, 16, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1),  # (64, 16, 28, 28)
            BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 16, 28, 28)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (192, 16, 14, 14)
            Mixed_3b(),  # (256, 32, 28, 28)
            Mixed_3c(),  # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # (480, 8, 7, 7)
            Mixed_4b(),  # (512, 16, 14, 14)
            Mixed_4c(),  # (512, 16, 14, 14)
            Mixed_4d(),  # (512, 16, 14, 14)
            Mixed_4e(),  # (528, 16, 14, 14)
            Mixed_4f(),  # (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),  # (832, 4, 3, 3)
            Mixed_5b(),  # (832, 8, 7, 7)
            Mixed_5c(),  # (1024, 8, 7, 7)
            # nn.AvgPool3d(kernel_size=(2, 3, 3), stride=1),  # (1024, 8, 1, 1)
            # nn.Dropout3d(dropout_keep_prob),
            # nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),  # (400, 8, 1, 1)
        )
        # self.spatial_squeeze = spatial_squeeze
        # self.softmax = nn.Softmax()
        self.out_feature = out_feature
        if out_feature == 'spatial_temporal_feature':
            self.drop_out = nn.Dropout(0.5)
            self.fcs = nn.ModuleList()
            for i in range(196):
                self.fcs.append(nn.Linear(1024, num_classes))
        elif out_feature == 'temporal_feature':
            self.drop_out = nn.Dropout(0.5)
            self.con_1x1 = nn.Conv1d(1024, num_classes, 1)
        else:
            self.drop_out = nn.Dropout(0.5)
            self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)

        N, C, T, H, W = x.shape
        if self.out_feature == 'spatial_temporal_feature':
            x = x.view(N, C, -1)  # 1,1024,4,7,7
            x = self.drop_out(x)
            xs = torch.chunk(x, T * H * W, dim=-1)
            x = []
            for i in range(T * H * W):
                x.append(self.fcs[i](xs[i].squeeze(-1)))  # N c
            x = torch.stack(x, dim=-1)
            return x  # N c 196
        if self.out_feature == 'spatial_feature':
            x = x.view(N, C, T, -1).mean(-2)  # 1,1024,4,7,7
            x = self.drop_out(x)
            x = self.con_1x1(x)
            return x
        x = x.view(x.size(0), 1024, -1)
        x = torch.mean(x, 2)  # N C T H W
        if self.out_feature == 'feature':
            return x
        x = self.drop_out(x)
        y = self.fc(x)
        # predictions = self.softmax(averaged_logits)
        return y
        # return predictions, averaged_logits

if __name__ == '__main__':

    import os
    from torch.autograd import Variable

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model = I3D(249)
    x = Variable(torch.zeros(1, 3, 128, 112, 112))
    r = model(x)
    print(r.shape)