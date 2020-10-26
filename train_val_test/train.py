# from __future__ import print_function, division

import os

os.environ['DISPLAY'] = 'localhost:10.0'
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../'])

import shutil
import time
import json
import torch
import setproctitle
# from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter
from tqdm import tqdm
from method_choose.data_choose import data_choose, init_seed
from method_choose.loss_choose import loss_choose
from method_choose.lr_scheduler_choose import lr_scheduler_choose
from method_choose.model_choose import model_choose
from method_choose.optimizer_choose import optimizer_choose
from method_choose.tra_val_choose import train_val_choose
from train_val_test import parser_args
from utility.log import TimerBlock, IteratorTimer
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict
import pickle


def rm_module(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


with TimerBlock("Good Luck") as block:
    # params
    args = parser_args.parser_args(block)

    init_seed(1)
    setproctitle.setproctitle(args.model_saved_name)

    block.log('work dir: ' + args.model_saved_name)
    if args.mode == 'train_val':
        train_writer = SummaryWriter(os.path.join(args.model_saved_name, 'train'), 'train')
        val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'val'), 'val')
    else:
        train_writer = val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'test'), 'test')

    global_step, start_epoch, model, optimizer_dict = model_choose(args, block)

    optimizer = optimizer_choose(model, args, val_writer, block)

    if optimizer_dict is not None and args.last_model is not None:
        try:
            optimizer.load_state_dict(optimizer_dict)
            block.log('load optimizer from state dict')
        except:
            block.log('optimizer not matched')
    else:
        block.log('no pretrained optimizer is loaded')

    loss_function = loss_choose(args, block)

    data_loader_train, data_loader_val = data_choose(args, block)

    lr_scheduler = lr_scheduler_choose(optimizer, args, start_epoch - 1, block)

    train_net, val_net = train_val_choose(args, block)

    best_accu = 0
    best_step = 0
    best_epoch = 0
    acc = 0
    loss = 100
    process = tqdm(range(start_epoch, args.max_epoch), 'Process: ' + args.model_saved_name)
    block.log('start epoch {} -> max epoch {}'.format(start_epoch, args.max_epoch))
    if args.val_first:
        model.eval()
        loss, acc, score_dict, all_pre_true, wrong_path_pre_true  = val_net(data_loader_val, model, loss_function, global_step, args, val_writer)
        block.log('Init ACC: {}'.format(acc))
    # lr = optimizer.param_groups[0]['lr']
    for epoch in process:
        last_epoch_time = time.time()
        model.train()  # Set model to training mode

        if args.lr_scheduler == 'reduce_by_epoch':
            lr_scheduler.step(epoch=epoch)
        elif args.lr_scheduler == 'reduce_by_acc':
            lr_scheduler.step(metric=acc, epoch=epoch)
        elif args.lr_scheduler == 'reduce_by_loss':
            lr_scheduler.step(metric=loss, epoch=epoch)
        else:
            lr_scheduler.step(epoch=epoch)
        lr = optimizer.param_groups[0]['lr']
        block.log('Current lr: {}'.format(lr))

        for key, value in model.named_parameters():
            value.requires_grad = True
        for freeze_key, freeze_epoch in args.freeze_keys:
            if freeze_epoch > epoch:
                block.log('{} is froze'.format(freeze_key))
                for key, value in model.named_parameters():
                    if freeze_key in key:
                        # block.log('{} is froze'.format(key))
                        value.requires_grad = False

        for lr_key, ratio_lr, ratio_wd, lr_epoch in args.lr_multi_keys:
            if lr_epoch > epoch:
                block.log('lr for {}: {}*{}, wd: {}*{}'.format(lr_key, lr, ratio_lr, args.wd, ratio_wd))
                for param in optimizer.param_groups:
                    if lr_key in param['key']:
                        param['lr'] *= ratio_lr
                        param['weight_decay'] *= ratio_wd

        global_step = train_net(data_loader_train, model, loss_function, optimizer, global_step, args, train_writer)
        block.log('Training finished for epoch {}'.format(epoch))
        model.eval()
        loss, acc, score_dict, all_pre_true, wrong_path_pre_true = val_net(data_loader_val, model, loss_function, global_step, args, val_writer)
        block.log('Validation finished for epoch {}'.format(epoch))

        if args.mode == 'train_val':
            train_writer.add_scalar('epoch', epoch, global_step)
            train_writer.add_scalar('lr', lr, global_step)
            train_writer.add_scalar('epoch_time', time.time() - last_epoch_time, global_step)

        if acc > best_accu:
            best_accu = acc
            best_step = global_step
            best_epoch = epoch
            save_score = args.model_saved_name + '/score.pkl'
            with open(save_score, 'wb') as f:
                pickle.dump(score_dict, f)
            with open(args.model_saved_name + '/all_pre_true.txt', 'w') as f:
                f.writelines(all_pre_true)
            with open(args.model_saved_name + '/wrong_path_pre_true.txt', 'w') as f:
                f.writelines(wrong_path_pre_true)

        # save model
        m = rm_module(model.state_dict())
        save = {
            'model': m,
            'optimizer': optimizer.state_dict()
        }
        torch.save(save, args.model_saved_name + '-latest.state')
        if (epoch + 1) % args.num_epoch_per_save == 0:
            torch.save(save, args.model_saved_name + '-' + str(epoch) + '-' + str(global_step) + '.state')

        process.set_description('Process: ' + args.model_saved_name + ' lr: ' + str(lr))
        block.log('EPOCH: {}, ACC: {:4f}, LOSS: {:4f}, EPOCH_TIME: {:4f}, LR: {}, BEST_ACC: {:4f}'
                  .format(epoch, acc, loss, time.time() - last_epoch_time, lr, best_accu))

        if lr < 1e-5:
            break

    m = rm_module(model.cpu().state_dict())
    save = {
        'model': m,
        'optimizer': optimizer.state_dict()
    }
    torch.save(save, args.model_saved_name + '-' + str(epoch) + '-' + str(global_step) + '.state')
    block.log(
        'Best model: ' + args.model_saved_name + '-' + str(best_epoch) + '-' + str(best_step) + '.state, acc: ' + str(
            best_accu))
    # block.save(os.path.join(args.model_saved_name, 'log.txt'))
