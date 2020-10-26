import torch
from utility.log import TimerBlock
from train_val_test.loss import L1, L2
import torch.nn.functional as func
import shutil
import inspect
import torch.nn as nn


# def to_onehot(num_class, label, alpha):
#     return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


# class naive_cross_entropy_loss(nn.Module):
#     def __init__(self, num_class, alpha):
#         self.num_class = num_class
#         self.alpha = alpha
#         super(naive_cross_entropy_loss, self).__init__()
#
#     def forward(self, inputs, target):
#         target = to_onehot(self.num_class, target, self.alpha)
#         return - (func.log_softmax(inputs, dim=-1) * target).sum(dim=-1).mean()


class multi_cross_entropy_loss(nn.Module):
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(size_average=True)
        super(multi_cross_entropy_loss, self).__init__()

    def forward(self, inputs, target):
        '''

            :param inputs: N C S
            :param target: N C
            :return: 
            '''
        num = inputs.shape[-1]
        inputs_splits = torch.chunk(inputs, num, dim=-1)
        loss = self.loss(inputs_splits[0].squeeze(-1), target)
        for i in range(1, num):
            loss += self.loss(inputs_splits[i].squeeze(-1), target)
        loss /= num
        return loss


def naive_cross_entropy_loss(inputs, target):
    return - (func.log_softmax(inputs, dim=-1) * target).sum(dim=-1).mean()


# def multi_cross_entropy_loss(inputs, target):
#     '''
#
#     :param inputs: N C S
#     :param target: N C
#     :return:
#     '''
#     num = inputs.shape[-1]
#     inputs_splits = torch.chunk(inputs, num, dim=-1)
#     loss = - (func.log_softmax(inputs_splits[0].squeeze(-1), dim=-1) * target).sum(dim=-1).mean()
#     for i in range(1, num):
#         loss += - (func.log_softmax(inputs_splits[i].squeeze(-1), dim=-1) * target).sum(dim=-1).mean()
#     loss /= num
#     return loss

# from warpctc_pytorch import CTCLoss
# class CTC(nn.Module):
#     def __init__(self, input_len, target_len):
#         super(CTC, self).__init__()
#         self.ctc = CTCLoss(size_average=True, length_average=False)  # TNC
#         self.input_len = input_len
#         self.target_len = target_len
#
#     def forward(self, input, target):
#         """
#         blank is default as 0 in ctc, but is -1 in model prob
#         :param input: TxNxcls
#         :param target: N, begin with 0
#         :return:
#         """
#         batch_size = target.shape[0]
#         input_ = torch.cat([input[:,:,-1:], input[:,:,:-1]], dim=-1).clone()
#         target_ = target + 1
#         input_.requires_grad_(True)
#         in_len = torch.IntTensor([self.input_len]*batch_size)  # .to(input_.get_device())
#         out_len = torch.IntTensor([self.target_len]*batch_size)  # .to(input_.get_device())
#         ls = self.ctc(input_.cpu(), target_.cpu(), in_len, out_len)
#         return ls


class CTC(nn.Module):
    def __init__(self, input_len, target_len, blank=0):
        super(CTC, self).__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
        self.input_len = input_len
        self.target_len = target_len

    def forward(self, input, target):
        """

        :param input: TxNxcls
        :param target: N
        :return:
        """
        batch_size = target.shape[0]
        input_ = torch.cat([input[:,:,-1:], input[:,:,:-1]], dim=-1).clone()
        target_ = target + 1
        target_ = target_.unsqueeze(-1)
        # target = torch.cat([target.unsqueeze(-1), target.unsqueeze(-1)], dim=1)
        ls = self.ctc(input_.log_softmax(2), target_, [self.input_len]*batch_size, [self.target_len]*batch_size)
        return ls


def loss_choose(args, block):
    loss = args.loss
    if loss == 'cross_entropy':
        # if args.mix_up_num > 0:
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)
        # else:
    elif loss == 'cross_entropy_naive':
        loss_function = naive_cross_entropy_loss
    elif loss == 'ctc':
        p = args.ls_param
        loss_function = CTC(p.input_len, p.target_len)
    elif loss == 'multi_cross_entropy':
        loss_function = multi_cross_entropy_loss()
    elif loss == 'mse_ce':
        loss_function = [torch.nn.MSELoss(), torch.nn.CrossEntropyLoss(size_average=True)]
    elif loss == 'l1loss':
        loss_function = L1()
    elif loss == 'l2loss':
        loss_function = L2()
    else:
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)

    block.log('Using loss: ' + loss)
    # shutil.copy2(inspect.getfile(loss_function), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)
    return loss_function


if __name__ == '__main__':
    res_ctc = torch.Tensor([[[0, 0, 1]], [[0.5, 0.6, 0.2]], [[0, 0., 1.]]])
    b = 1  # batch
    c = 2  # label有多少类
    in_len = 3  # 预测每个序列有多少个label
    label_len = 1  # 实际每个序列有多少个label

    # res_ctc = torch.rand([in_len, b, c+1])
    target = torch.zeros([b*label_len,], dtype=torch.long)

    loss_ctc = CTC(in_len, label_len)
    ls_ctc = loss_ctc(res_ctc, target)

    # loss_ctcp = CTCP(in_len, label_len)
    # ls_ctcp = loss_ctcp(res_ctc, target)

    # print(ls_ctc, ls_ctcp)
