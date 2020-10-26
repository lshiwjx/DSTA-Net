import torch
import os

os.environ['DISPLAY'] = 'localhost:10.0'
import matplotlib.pyplot as plt
import numpy as np


def draw_skeleton2(ax, joints, title='', axis=False, connect=None, additional_connect=None, zlimb=[0, 1], label="",
                   azim=65, elev=-10):
    colors = ['dodgerblue', 'steelblue', 'darkorange']
    # plot joint
    for num, joint in enumerate(joints):
        ax.scatter(joint[0], joint[1], joint[2], color=colors[0])
    # plot edge
    if connect is None:
        connectivity = ((0, 1),
                        (1, 2), (2, 3), (3, 4), (4, 5),
                        (1, 6), (6, 7), (7, 8), (8, 9),
                        (1, 10), (10, 11), (11, 12), (12, 13),
                        (1, 14), (14, 15), (15, 16), (16, 17),
                        (1, 18), (18, 19), (19, 20), (20, 21))
    else:
        connectivity = connect

    for connection in connectivity:
        t = connection[0]
        f = connection[1]
        ax.plot([joints[f][0], joints[t][0]], [joints[f][1], joints[t][1]], [joints[f][2], joints[t][2]],
                color=colors[1], linewidth=3)

    # plot additional edges
    if not additional_connect is None:
        for connection in additional_connect:
            t = connection[0]
            f = connection[1]
            s = connection[2]
            ax.plot([joints[f][0], joints[t][0]], [joints[f][1], joints[t][1]], [joints[f][2], joints[t][2]],
                    color=colors[2], alpha=s, linewidth=2)

    ax.view_init(azim=azim, elev=-elev)

    # ax.set_title(title)
    if not axis:
        ax.set_axis_off()

        # 放置文字，以0-1为准（trainform的原因）
        # ax.text2D(0.3, -0.03, title, transform=ax.transAxes)
        # ax.text2D(-0.12, 0.6, label, transform=ax.transAxes, rotation=90)

        # 设置各个轴的视野
        # plt.xlim([-0.4, 0.4])
        # ax.set_zlim(zlimb[0], zlimb[1])
        # plt.ylim([-0.2, 0.4])

        # 设置整张图像的边框
        # params = dict(bottom=0.05, left=0.09, right=1, top=1)
        # fig.subplots_adjust(**params)


def get_connection_from_attention(a, r=0.1):
    connect0 = np.abs(a) ** 2
    connect0 -= connect0.min()
    connect0 /= connect0.max()
    num = connect0.shape[0] * connect0.shape[1]
    connect = connect0 > np.sort(connect0.reshape(num))[-int(num * r)]
    cs = []
    for i in range(connect.shape[0]):
        for j in range(connect.shape[1]):
            if connect[i, j]:
                cs.append([int(i), int(j), connect0[i, j]])
    return cs

model = torch.load('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/shrec/stagcnattention_drop02_1004050_128_att02s_att02t_agcnconfig_newpet-119-7320.state')
# model = torch.load(
#     '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/dhg/stagcnattention_drop02_1004050_8_att02s_att02t_warm5_init1_nog_tan_lrelu-129-7930.state')
# model = torch.load('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/ntu/ntulshifullbody_agcnattention-99-125200.state')
# model=torch.load('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/dhg28/stagcnattention_drop02_1508020_8_att01_warm5_28-171-10492.state')
# model = torch.load('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/fpha/agcnattention_drop02_1508020_8_att01_warm5-175-3168.state')
# model = torch.load('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/jhmdb/stagcnattention_drop0_reduce_32_att01_warm5_C2_s1_joint-89-1800.state')
beta = []
gamma = []
PA = []
alpha = []
for key, value in model['model'].items():
    if 'alpha' in key:
        alpha.append(value.cpu().numpy())
    if 'beta' in key:
        beta.append(value.cpu().numpy())
    if 'gamma' in key:
        gamma.append(value.cpu().numpy())
    if 'attention0' in key:
        PA.append(value.cpu().numpy())

import pickle
from mpl_toolkits.mplot3d import Axes3D

s_ratio = [np.abs(alpha[0]), np.abs(gamma[0])]
t_ratio = [np.abs(alpha[1]), np.abs(gamma[1])]
xticks = ['spatial', 'temporal']
# xticks = ['att2', 'att1', 'att0']
bar_width = 1
subset = 0
ratios = [[float(s_ratio[i][0, subset, 0, 0]), float(t_ratio[i][0, subset, 0, 0])] for i in range(2)]
ratiopre = ratios[0]
ratiomid = ratios[1]
# ratiopre = [float(s / (s + t)) for s, t in zip(s_ratio, t_ratio)]
# ratioaft = [float(t / (s + t)) for s, t in zip(s_ratio, t_ratio)]
# positions of the left bar-boundaries
bar_l = [i for i in range(len(ratiopre))]
# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i + (bar_width / 2) for i in bar_l]
# Create the total score for each participant
totals = [i + j for i, j in zip(ratiopre, ratiomid)]
# Create the percentage of the total score the pre_score value for each participant was
pre_rel = [i / j * 100 for i, j in zip(ratiopre, totals)]
# Create the percentage of the total score the mid_score value for each participant was
mid_rel = [i / j * 100 for i, j in zip(ratiomid, totals)]

f, ax = plt.subplots(1, figsize=(4, 5))
pre = ax.bar(bar_l,
             # using pre_rel data
             pre_rel,
             # labeled
             label='Pre Score',
             # with alpha
             alpha=0.9,
             # with color
             color='#019600',
             # with bar width
             width=bar_width,
             # with border color
             edgecolor='white'
             )
mid = ax.bar(bar_l,
             # using mid_rel data
             mid_rel,
             # with pre_rel
             bottom=pre_rel,
             # labeled
             label='Mid Score',
             # with alpha
             alpha=0.9,
             # with color
             color='#3C5F5A',
             # with bar width
             width=bar_width,
             # with border color
             edgecolor='white'
             )

# Set the ticks to be first names
plt.xticks(tick_pos, xticks)
ax.set_ylabel("Percentage")
ax.set_xlabel("")
# Let the borders of the graphic
# plt.xlim([min(tick_pos) - bar_width, max(tick_pos) + bar_width])
plt.ylim(0, 110)
# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='center')
plt.legend((pre[0], mid[0]), ('att2', 'att0'), loc=0, ncol=5)
plt.tight_layout()
plt.show()
# outdir = '/home/lshi/Project/Pytorch/st-gcn/work_dir/VIS/ntu_mystgcn_attentions25t9c2nfalpha_adaptivePA5tanhalpha_SAT-49-29400/graphs/'
# outname = 'proportion'
# plt.savefig(outdir + outname + ".pdf", format='pdf', bbox_inches='tight')
print(beta)
