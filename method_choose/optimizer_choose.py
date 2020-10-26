from __future__ import print_function, division

import torch
from utility.log import TimerBlock
from torch.optim.sgd import SGD
import shutil
import inspect


def optimizer_choose(model, args, writer, block):
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{'params': [value], 'lr': args.lr, 'key': key, 'weight_decay': args.wd}]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params)
        block.log('Using Adam optimizer')
    elif args.optimizer == 'sgd':
        momentum = 0.9
        optimizer = SGD(params, momentum=momentum)
        block.log('Using SGD with momentum ' + str(momentum))
    elif args.optimizer == 'sgd_nev':
        momentum = 0.9
        optimizer = SGD(params, momentum=momentum, nesterov=True)
        block.log('Using SGD with momentum ' + str(momentum) + 'and nesterov')
    else:
        momentum = 0.9
        optimizer = SGD(params, momentum=momentum)
        block.log('Using SGD with momentum ' + str(momentum))

    # shutil.copy2(inspect.getfile(optimizer), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)
    return optimizer
