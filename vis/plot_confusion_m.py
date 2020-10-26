import os

# 纵坐标代表原来的类别，横坐标代表预测成的类别
os.environ['DISPLAY'] = 'localhost:10.0'
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


def get_m(f):
    lines = f.readlines()
    pre_list = []
    true_list = []
    for line in lines:
        pre, true = line[:-1].split(',')
        pre_list.append(int(pre))
        true_list.append(int(true))
    m = confusion_matrix(true_list, pre_list)
    return m


def plot_m(ax, m):
    norm_conf = []
    # for i in m:
    #     a = 0
    #     tmp_arr = []
    #     a = sum(i, 0)
    #     for j in i:
    #         tmp_arr.append(float(j) / float(a))
    #     norm_conf.append(tmp_arr)
    ax.set_aspect(1)
    res = ax.imshow(np.array(m), cmap='YlGn',
                    interpolation='nearest')
    plt.gca().invert_yaxis()
    width, height = m.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(m[x][y], 2)), xy=(y, x), size=8,
                        horizontalalignment='center',
                        verticalalignment='center')

    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # plt.xticks(range(width), alphabet[:width])
    # plt.yticks(range(height), alphabet[:height])
    # plt.savefig('confusion_matrix.png', format='png')
    return res


import matplotlib

label = 0
layer = 0
num_layer = 8
num_joints = 22
subset = 2
font = {'family': 'normal',
        'size': 10}
matplotlib.rc('font', **font)

label_file = open("../prepare/dhg/dha_label.txt")
classes = label_file.readlines()
classes = [x[:-1] for x in classes]
label_txt = classes[label]

joint_file = open("../prepare/dhg/dhg_joints.txt")
joints = joint_file.readlines()
joints = [x[:-1] for x in joints]
tick_marks = np.arange(len(joints))

fig = plt.figure(figsize=[16, 8])
# plt.title(label_txt)
att = np.zeros([num_layer, num_joints, num_joints])
for l in range(num_layer):
    f = np.load(
        '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/val/shrec/s{}{}.npy'.format(label, l))
    att[l] += f[0].mean(0)
# att = np.zeros([subset, num_joints, num_joints])
# for l in range(num_layer):
#     f = np.load(
#         '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/val/shrec/s{}{}.npy'.format(label, l))
#     att[l] += f[0].mean(0)
layers = [5, 6]
for s in range(subset):
    ax = plt.subplot(1, subset, s + 1)
    m = att[layers[s]]
    m -= m.min()
    m /= m.max()
    m -= 0.01
    res1 = plot_m(ax, m)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(joints, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(joints)
    ax.set_title('Layer_'+str(layers[s]+1))
    # fig.colorbar(res1)

# tick_marks = np.arange(len(joints))
# plt.xticks(tick_marks, joints, rotation=90)
# plt.yticks(tick_marks, joints, size=10)
plt.tight_layout()
# plt.show()
outdir = '/home/lshi/Project/Pytorch/3d-resnet-pytorch/vis/agcnv2/'
outname = 'attentionmap67'
plt.savefig(outdir + outname + ".pdf", format='pdf', bbox_inches='tight')