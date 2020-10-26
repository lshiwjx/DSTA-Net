import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--label', default='/home/lshi/Database/ntu_60/xsub/val_label.pkl',
                    help='')
parser.add_argument('--spatial_temporal', default='./work_dir/ntu60/dstanet_drop0_6090120_128_ST')
parser.add_argument('--spatial', default='./work_dir/ntu60/dstanet_drop0_6090120_128_S')
parser.add_argument('--temporal_slow', default='./work_dir/ntu60/dstanet_drop0_6090120_128_T1')
parser.add_argument('--temporal_fast', default='./work_dir/ntu60/i3d_101520')
parser.add_argument('--alpha', default=[1, 1, 1, 3], help='weighted summation')
arg = parser.parse_args()

label = open(arg.label, 'rb')
label = np.array(pickle.load(label))
r1 = open('{}/score.pkl'.format(arg.spatial_temporal), 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('{}/score.pkl'.format(arg.spatial), 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('{}/score.pkl'.format(arg.temporal_slow), 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('{}/score.pkl'.format(arg.temporal_fast), 'rb')
r4 = list(pickle.load(r4).items())
right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
