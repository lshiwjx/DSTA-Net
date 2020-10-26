'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss

import torch
import torch.nn as nn
import math


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return ['L1', 'EPE'], [lossvalue, epevalue]


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return ['L2', 'EPE'], [lossvalue, epevalue]

