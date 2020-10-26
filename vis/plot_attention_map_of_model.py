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
    """

    :param a: NxN
    :param r:
    :return:
    """
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
PAS = []
PAT = []
alpha = []
for key, value in model['model'].items():
    if 'alpha' in key:
        alpha.append(value.cpu().numpy())
    if 'beta' in key:
        beta.append(value.cpu().numpy())
    if 'gamma' in key:
        gamma.append(value.cpu().numpy())
    if 'attention0s' in key:
        PAS.append(value.cpu().numpy())
    if 'attention0t' in key:
        PAT.append(value.cpu().numpy())

import pickle
from mpl_toolkits.mplot3d import Axes3D

examples = pickle.load(open('/home/lshi/Database/dhg_hand_shrec/val_skeleton_ddnet.pkl', 'rb'))  # NCTVM
example_joints = examples[4][:, 7, :, 0].transpose(1, 0)  # cv->vc
# find examples
# for i in range(4, 100):
#     example_joints = examples[i][:, 7, :, 0].transpose(1, 0)  # cv->vc
#     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), subplot_kw=dict(projection='3d'))
#     draw_skeleton2(axes, example_joints, title='', axis=False, azim=-75, elev=60)
#     plt.show()

# plot spatial attention
fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(16, 4), subplot_kw=dict(projection='3d'))
a_s = PAS
subset = 0
for i, a in enumerate(a_s):
    a = a[0, subset]
    cs = get_connection_from_attention(a, r=0.05)
    ax = axes[i]
    draw_skeleton2(ax, example_joints, additional_connect=cs, title='',
                   axis=False, azim=-150, elev=60)
plt.show()

# plot temporal attention
a_t = PAT
fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(16, 4))
for i, a in enumerate(a_t):
    a = a[0, subset]
    a -= a.min()
    a /= a.max()
    a *= 255
    axes[i].imshow(a.astype(np.uint8), cmap='gray')
plt.show()