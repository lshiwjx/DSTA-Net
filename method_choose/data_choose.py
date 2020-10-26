from __future__ import print_function, division

# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import numpy as np
import random
import shutil
import inspect
from dataset.ntu_skeleton import NTU_SKE
from dataset.fpha_skeleton import FPHA_SKE
from dataset.dhg_skeleton import DHG_SKE
from dataset.rgb import RGB
from dataset.rgb_gulp import RGBGULP
from easydict import EasyDict as edict


def init_seed(x):
    # pass
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_choose(args, block):
    if args.mode == 'test' or args.mode == 'watch_off':
        if args.data == 'ntu_skeleton':
            workers = args.worker
            data_set_val = NTU_SKE(mode='eval_rot', **args.data_param['val_data_param'])
        else:
            raise (RuntimeError('No data loader'))
        data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=workers, drop_last=False, pin_memory=args.pin_memory,
                                     worker_init_fn=init_seed)
        data_loader_train = None

        block.log('Data load finished: ' + args.data)

        shutil.copy2(__file__, args.model_saved_name)
        return data_loader_train, data_loader_val
    else:
        if args.data == 'ntu_skeleton':
            workers = args.worker
            data_set_train = NTU_SKE(mode='train', **args.data_param['train_data_param'])
            data_set_val = NTU_SKE(mode='val', **args.data_param['val_data_param'])
        elif args.data == 'fpha_skeleton':
            workers = args.worker
            data_set_train = FPHA_SKE(mode='train', **args.data_param['train_data_param'])
            data_set_val = FPHA_SKE(mode='val', **args.data_param['val_data_param'])
        elif args.data == 'dhg_skeleton':
            workers = args.worker
            data_set_train = DHG_SKE(mode='train', **args.data_param['train_data_param'])
            data_set_val = DHG_SKE(mode='val', **args.data_param['val_data_param'])
        elif args.data == 'shrec_skeleton':
            workers = args.worker
            data_set_train = DHG_SKE(mode='train', **args.data_param['train_data_param'])
            data_set_val = DHG_SKE(mode='val', **args.data_param['val_data_param'])
        elif args.data == 'rgb':
            workers = args.worker
            data_set_train = RGB(mode='train', arg=edict(args.data_param['train_data_param']))
            data_set_val = RGB(mode='val', arg=edict(args.data_param['val_data_param']))
        elif args.data == 'rgbgulp':
            workers = args.worker
            data_set_train = RGBGULP(mode='train', arg=edict(args.data_param['train_data_param']))
            data_set_val = RGBGULP(mode='val', arg=edict(args.data_param['val_data_param']))
        else:
            raise (RuntimeError('No data loader'))
        data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=workers, drop_last=False, pin_memory=args.pin_memory,
                                     worker_init_fn=init_seed)
        data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=workers, drop_last=True, pin_memory=args.pin_memory,
                                       worker_init_fn=init_seed)

        block.log('Data load finished: ' + args.data)

        shutil.copy2(__file__, args.model_saved_name)
        return data_loader_train, data_loader_val
