import torch
import torch.nn as nn
import math
import numpy as np


# def fix_t(t):
#     a1 = np.eye(t)
#     a2 = np.zeros([t, t])
#     a3 = np.zeros([t, t])
#     for i in range(t):
#         for j in range(t):
#             if i - j == 1:
#                 a2[i, j] = 1
#             if j - i == 1:
#                 a3[i, j] = 1
#     return np.stack((a1, a2, a3)).astype(np.float32)


def fix_t(t, num_sub):
    mtxs = [np.zeros([t, t]) for _ in range(num_sub)]
    for i in range(t):
        for j in range(t):
            for k in range(num_sub):
                dis = k - num_sub // 2
                if i - j == dis:
                    mtxs[k][i, j] = 1
    return np.stack(mtxs).astype(np.float32)


def conv_init(conv):
    # nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.weight, 1)
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    # nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.weight, 1)
    nn.init.constant_(fc.bias, 0)


class ElementBn(nn.Module):
    def __init__(self, num_node, channels):
        super(ElementBn, self).__init__()
        self.bn = nn.BatchNorm1d(num_node * channels)

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.bn(x).view(N, C, V, T).permute(0, 1, 3, 2)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        self.b_2 = nn.Parameter(torch.zeros(channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        self.eps = eps

    def forward(self, x):
        # NCTV
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal" or domain == "mask_t":
            # temporal positial embedding
            # pos_list = list(range(self.joint_num * self.time_len))
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial" or domain == "mask_s":
            # spatial positial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)
        # position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, att0s=True, att1s=True, att2s=True, att0t=True, att1t=True, att2t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.att0s = att0s
        self.att1s = att1s
        self.att2s = att2s
        self.att0t = att0t
        self.att1t = att1t
        self.att2t = att2t

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att2s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            if att1s:
                self.x1_nets = nn.Parameter(
                    torch.rand(num_subset, in_channels, num_frame, num_node) / inter_channels / num_frame)
                self.betas = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            if att0s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node)
                self.gammas = nn.Parameter(torch.ones(1, num_subset, 1, 1))

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att2t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            if att1t:
                self.x1_nett = nn.Parameter(
                    torch.rand(num_subset, out_channels, num_frame, num_node) / inter_channels / num_node)
                self.betat = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            if att0t:
                # self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame))
                attention0t = torch.from_numpy(fix_t(num_frame, num_subset)[np.newaxis, ...])
                self.register_buffer('attention0t', attention0t)
                self.gammat = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.att2s:
                q, k = torch.chunk(self.in_nets(self.pes(x)).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', (q, k)) / (self.inter_channels * T)) * self.alphas
            if self.att1s:
                attention = attention + self.tan(torch.einsum('sctu,nctv->nsuv', (self.x1_nets, x))) * self.betas
            if self.att0s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', (x, attention)).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.use_temporal_att:
            attention = self.attt
            if self.att2t:
                q, k = torch.chunk(self.in_nett(self.pet(y)).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', (q, k)) / (self.inter_channels * V)) * self.alphat
            if self.att1t:
                attention = attention + self.tan(torch.einsum('sctv,ncqv->nstq', (self.x1_nett, y))) * self.betat
            if self.att0t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', (y, attention)).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z


class AGcnAttention(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, num_person=2, dropout=0., config=None,
                 num_channel=3, att0s=True, att1s=False, att2s=True, att0t=False, att1t=False, att2t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0):
        super(AGcnAttention, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'att0s': att0s,
            'att1s': att1s,
            'att2s': att2s,
            'att0t': att0t,
            'att1t': att1t,
            'att2t': att2t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'attentiondrop': attentiondrop
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Linear(self.out_channels, num_class)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = self.input_map(x)

        for m in self.graph_layers:
            x = m(x)
        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)  # whole spatial of one channel

        return self.fc(x)

if __name__ == '__main__':
    from model.dstanet import DSTANet
    from method_choose.data_choose import init_seed
    init_seed(1)
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    net = DSTANet(config=config)  # .cuda()
    net2 = AGcnAttention(config=config)
    ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    out1 = net(ske)
    out2 = net2(ske)
    print(net(ske).shape)