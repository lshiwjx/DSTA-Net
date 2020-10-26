from __future__ import print_function, division
import torch
import torch.nn as nn
from collections import OrderedDict
import shutil
import inspect
from model.dstanet import DSTANet


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


def model_choose(args, block):
    m = args.model
    if m == 'dstanet':
        model = DSTANet(num_class=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(DSTANet), args.model_saved_name)
    else:
        raise (RuntimeError("No modules"))

    shutil.copy2(__file__, args.model_saved_name)
    block.log('Model load finished: ' + args.model + ' mode: train')
    optimizer_dict = None

    if args.pre_trained_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pre_trained_model)  # ['state_dict']
        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = rm_module(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys = list(pretrained_dict.keys())
        for key in keys:
            for weight in args.ignore_weights:
                if weight in key:
                    if pretrained_dict.pop(key) is not None:
                        block.log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        block.log('Can Not Remove Weights: {}.'.format(key))
        block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        # block.log(model_dict)
        model.load_state_dict(model_dict)
        block.log('Pretrained model load finished: ' + args.pre_trained_model)

    global_step = 0
    global_epoch = 0
    # The name for model must be **_**-$(step).state
    if args.last_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.last_model)  # ['state_dict']
        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = rm_module(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        block.log('In last model, following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        try:
            global_step = int(args.last_model[:-6].split('-')[2])
            global_epoch = int(args.last_model[:-6].split('-')[1])
        except:
            global_epoch = global_step = 0
        block.log('Training continue, last model load finished, step is {}, epoch is {}'.format(str(global_step),
                                                                                                str(global_epoch)))

    model.cuda()
    model = nn.DataParallel(model, device_ids=args.device_id)
    block.log('copy model to gpu')
    return global_step, global_epoch, model, optimizer_dict
