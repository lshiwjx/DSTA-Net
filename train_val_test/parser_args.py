import argparse
import os
from utility.log import TimerBlock
import colorama
import torch
import shutil
import yaml
from easydict import EasyDict as ed


def parser_args(block):
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='')
    parser.add_argument('-model', default='resnet3d_50')
    parser.add_argument('-model_param', default={}, help=None)
    # classify_multi_crop classify classify_pose
    parser.add_argument('-train', default='classify')
    parser.add_argument('-val_first', default=False)
    parser.add_argument('-data', default='jmdbgulp')
    parser.add_argument('-data_param', default={}, help='')
    # train_val test train_test
    parser.add_argument('-mode', default='train_val')
    # cross_entropy mse_ce
    parser.add_argument('-loss', default='cross_entropy')
    parser.add_argument('-ls_param', default={
    })
    # reduce_by_acc reduce_by_loss reduce_by_epoch cosine_annealing_lr
    parser.add_argument('-lr_scheduler', default='reduce_by_acc')
    parser.add_argument('-lr_param', default={})
    parser.add_argument('-warm_up_epoch', default=0)
    parser.add_argument('-step', default=[80, ])
    parser.add_argument('-lr', default=0.01)  # 0.001
    parser.add_argument('-wd', default=1e-4)  # 5e-4
    parser.add_argument('-lr_decay_ratio', default=0.1)
    parser.add_argument('-lr_multi_keys', default=[
        ['fc', 1, 1, 0], ['bn', 1, 1, 0],
    ], help='key, lr ratio, wd ratio, epoch')
    parser.add_argument('-optimizer', default='sgd_nev')
    parser.add_argument('-freeze_keys', default=[
        ['PA', 5],
    ], help='key, epoch')

    parser.add_argument('-class_num', default=12)
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-worker', default=16)
    parser.add_argument('-pin_memory', default=False)
    parser.add_argument('-max_epoch', default=50)

    parser.add_argument('-num_epoch_per_save', default=2)
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-last_model', default=None, help='')
    parser.add_argument('-ignore_weights', default=['fc'])
    parser.add_argument('-pre_trained_model', default='')
    parser.add_argument('--label_smoothing_num', default=0, help='0-1: 0 denotes no smoothing')
    parser.add_argument('--mix_up_num', default=0, help='0-1: 1 denotes uniform distribution, smaller, more concave')
    parser.add_argument('-device_id', default=[0, 1, 2, 3])
    parser.add_argument('-debug', default=False)
    parser.add_argument('-cuda_visible_device', default='0, 1, 2, 3, 4, 5, 6, 7')
    parser.add_argument('-grad_clip', default=0)
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_device

    if args.debug:
        args.device_id = [0]
        args.batch_size = 1
        args.worker = 0
        os.environ['DISPLAY'] = 'localhost:10.0'
    block.addr = os.path.join(args.model_saved_name, 'log.txt')

    if os.path.isdir(args.model_saved_name) and not args.last_model and not args.debug:
        print('log_dir: ' + args.model_saved_name + ' already exist')
        answer = input('delete it? y/n:')
        if answer == 'y':
            shutil.rmtree(args.model_saved_name)
            print('Dir removed: ' + args.model_saved_name)
            input('refresh it')
        else:
            print('Dir not removed: ' + args.model_saved_name)

    if not os.path.exists(args.model_saved_name):
        os.makedirs(args.model_saved_name)
    # Get argument defaults (has tag #this is a hack)
    parser.add_argument('--IGNORE', action='store_true')
    # 会返回列表
    defaults = vars(parser.parse_args(['--IGNORE']))
    # Print all arguments, color the non-defaults
    for argument, value in sorted(vars(args).items()):
        reset = colorama.Style.RESET_ALL
        color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        block.log('{}{}: {}{}'.format(color, argument, value, reset))

    shutil.copy2(__file__, args.model_saved_name)
    shutil.copy2(args.config, args.model_saved_name)

    args = ed(vars(args))
    return args
