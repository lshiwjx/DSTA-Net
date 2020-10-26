from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from utility.log import TimerBlock
from torch.optim.lr_scheduler import _LRScheduler
import shutil
import inspect


class GradualWarmupScheduler():
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, after_scheduler=None, last_epoch=-1):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        # 提供epoch0的初始学习率，改掉继承optimizer时继承的
        # if type(after_scheduler) is ReduceLROnPlateau:
        #     print(type(after_scheduler))
        #     if after_scheduler.mode=='min':
        #         init_metric = 100
        #     else:
        #         init_metric = 0
        #     self.step(metric=init_metric, epoch=last_epoch+1)
        # else:
        #     self.step(epoch=last_epoch+1)
        # super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            # return super(GradualWarmupScheduler, self).step(epoch)
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr


def lr_scheduler_choose(optimizer, args, last_epoch, block):
    lr_args = args.lr_param
    if args.lr_scheduler == 'reduce_by_acc':
        lr_patience = lr_args['lr_patience']
        lr_threshold = lr_args['lr_threshold']
        lr_delay = lr_args['lr_delay']
        block.log('lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
                  .format(args.lr, args.lr_decay_ratio, lr_patience, lr_threshold, lr_delay))
        lr_scheduler_pre = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_ratio,
                                             patience=lr_patience, verbose=True,
                                             threshold=lr_threshold, threshold_mode='abs',
                                             cooldown=lr_delay)
        lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
                                              after_scheduler=lr_scheduler_pre, last_epoch=last_epoch)
    elif args.lr_scheduler == 'reduce_by_loss':
        lr_patience = lr_args['lr_patience']
        lr_threshold = lr_args['lr_threshold']
        lr_delay = lr_args['lr_delay']
        block.log('lr scheduler: lr: {} DecayRatio: {} Patience: {} Threshold: {} Before_epoch: {}'
                  .format(args.lr, args.lr_decay_ratio, lr_patience, lr_threshold, lr_delay))
        lr_scheduler_pre = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_ratio,
                                             patience=lr_patience, verbose=True,
                                             threshold=lr_threshold, threshold_mode='abs',
                                             cooldown=lr_delay)
        lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
                                              after_scheduler=lr_scheduler_pre, last_epoch=last_epoch)
    elif args.lr_scheduler == 'reduce_by_epoch':
        step = lr_args['step']
        block.log('lr scheduler: Reduce by epoch, step: ' + str(step))
        lr_scheduler_pre = MultiStepLR(optimizer, step, last_epoch=last_epoch, gamma=args.lr_decay_ratio)
        lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
                                              after_scheduler=lr_scheduler_pre, last_epoch=last_epoch)
    elif args.lr_scheduler == 'cosine_annealing_lr':
        lr_scheduler_pre = CosineAnnealingLR(optimizer, lr_args.max_epoch + 1, eta_min=0.0001, last_epoch=last_epoch)
        lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
                                              after_scheduler=lr_scheduler_pre, last_epoch=last_epoch)
    else:
        raise ValueError()
    # shutil.copy2(inspect.getfile(lr_scheduler), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)
    return lr_scheduler
