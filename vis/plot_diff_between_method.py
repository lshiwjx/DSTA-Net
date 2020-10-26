import os

os.environ['DISPLAY'] = 'localhost:10.0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def acc_of_each_cls_from_score(score, label, num_cls):
    right_nums = [0 for _ in range(num_cls)]
    total_nums = [0 for _ in range(num_cls)]
    for i in range(len(label[0])):
        _, l = label[:, i]
        _, s = score[i]
        r = np.argmax(s)
        right_nums[int(l)] += int(r == int(l))
        total_nums[int(l)] += 1
    accs = [x/y for x,y in zip(right_nums, total_nums)]
    print('total acc: ', sum(accs)/num_cls)
    return accs

#
# def get_m(f):
#     lines = f.readlines()
#     pre_list = []
#     true_list = []
#     for line in lines:
#         pre, true = line[:-1].split(',')
#         pre_list.append(int(pre))
#         true_list.append(int(true))
#     m = confusion_matrix(true_list, pre_list)
#     return m

import pickle
import matplotlib

font = {
    # 'family': 'aria',
        'size': 10}
matplotlib.rc('font', **font)
label = open('/home/lshi/Database/dhg_hand_shrec/val_label_ddnet_14.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(
    '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/shrec/stagcnattention_drop02_1004050_128_att02s_att2t_agcnconfig_newpet/score.pkl',
    'rb')
r1 = list(pickle.load(r1).items())
r2 = open(
    '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/shrec/stagcnattention_drop02_6090120_128_att02s_att2t_agcnconfig_newpet_bone/score.pkl',
    'rb')
r2 = list(pickle.load(r2).items())
r3 = open(
    '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/shrec/stagcnattention_drop02_6090120_128_att02s_att2t_agcnconfig_newpet_v1/score.pkl',
    'rb')
r3 = list(pickle.load(r3).items())
r4 = open(
    '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/shrec/stagcnattention_drop02_6090120_128_att02s_att2t_agcnconfig_newpet_v2/score.pkl',
    'rb')
r4 = list(pickle.load(r4).items())
num_class = 14
a1 = acc_of_each_cls_from_score(r1, label, num_class)
a2 = acc_of_each_cls_from_score(r2, label, num_class)
a3 = acc_of_each_cls_from_score(r3, label, num_class)
a4 = acc_of_each_cls_from_score(r4, label, num_class)
# total = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_c3d_kpre_32f_valt.txt')
# total_regular = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# total_reverse = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# total_single = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# m_reg = get_m(total_regular)
# m_rev = get_m(total_reverse)
# m_sin = get_m(total_single)
plt.figure(figsize=[12.5,3.5])
# plt.title('Recognition Accuracy for each classes')
plt.ylabel('acc')
# reg = []
# for index, line in enumerate(m_reg):
#     reg.append(line[index] / sum(line))
# rev = []
# for index, line in enumerate(m_rev):
#     rev.append(line[index] / sum(line))
# sin = []
# for index, line in enumerate(m_sin):
#     sin.append(line[index] / sum(line))
index = np.arange(len(a1))
bar_width = 0.3
rects_reg = plt.bar(index - bar_width, a2, bar_width, color='#4bacc6')
rects_rev = plt.bar(index, a3, bar_width, color='#4f81bd')
rects_sin = plt.bar(index + bar_width, a4, bar_width, color='#8064a2')

diff_reg_rev = [a2[i] - a3[i] for i in range(num_class)]
diff_reg_sin = [a3[i] - a4[i] for i in range(num_class)]
diff_line_rev, = plt.plot(diff_reg_rev, color='orange', linestyle='-.')
diff_line_sin, = plt.plot(diff_reg_sin, color='#ff007f', linestyle='-.')
label_file = open("/home/lshi/Database/dhg_hand_shrec/labels.txt")
classes = label_file.readlines()
classes = [x[:-1] for x in classes]
tick_marks = np.arange(len(classes))
plt.ylim([-0.2, 1.2])
plt.xticks(tick_marks, classes, rotation=45)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['right'].set_visible(False)
plt.legend((rects_reg[0], rects_rev[0], rects_sin[0], diff_line_rev, diff_line_sin),
           ('spatial', 'slow_temporal', 'fast_temporal', 'spatial - slow_temporal', 'slow_temporal - fast_temporal'), loc=0, ncol=5)
# plt.show()
plt.savefig('../../vis/agcnv2/featurefusionacc.pdf', format='pdf', bbox_inches='tight')