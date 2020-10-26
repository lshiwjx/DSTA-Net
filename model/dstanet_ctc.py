import torch
import torch.nn as nn
import math
import numpy as np


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

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
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

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
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
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

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
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
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z


class DSTANetCTC(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(DSTANetCTC, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Conv1d(self.out_channels, num_class, 1)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        self.soft = nn.Softmax(dim=1)

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

        for i, m in enumerate(self.graph_layers):
            x = m(x)

        # NM, C, T, V -> N, C, T
        x = x.view(N, M, self.out_channels, T//4, V).mean(1).mean(-1)

        x = self.fc(x)  # N, C, T -> N, cls, T

        # x = self.soft(x)

        return x.permute(2, 0, 1)  # T, N, cls


if __name__ == '__main__':
    # config = [[64, 64, 16, 1], [64, 64, 16, 1],
    #           [64, 128, 32, 2], [128, 128, 32, 1],
    #           [128, 256, 64, 2], [256, 256, 64, 1],
    #           [256, 256, 64, 1], [256, 256, 64, 1],
    #           ]
    # net = DSTANetCTC(config=config, num_class=61)  # .cuda()
    # ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    # res = net(ske)
    # print(res.shape)

    res_ctc = torch.Tensor([[[0, 0, 1]], [[0, 0, 1]], [[0., 0., 1]]])  # 3x1x3
    res_ce = torch.Tensor([[0, 0., 1]])
    loss_ctc = nn.CTCLoss(blank=2, reduction='mean')
    loss_ce = nn.CrossEntropyLoss()
    target = torch.ones([1, 1], dtype=torch.long)
    input_len = (3,)
    target_len = (1,)

    ls_ctc = loss_ctc(res_ctc.log_softmax(2), target, input_len, target_len)
    ls_ce = loss_ce(res_ce, target.squeeze(-1))

    print(ls_ctc, ls_ce)

