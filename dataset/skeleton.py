import os
import sys
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
from dataset.video_data import *


class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,
                 mode='train', decouple_spatial=False, num_skip_frame=None,
                 random_choose=False, center_choose=False):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.num_skip_frame = num_skip_frame
        self.decouple_spatial = decouple_spatial
        self.edge = None
        self.load_data()

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = int(self.label[index])
        sample_name = self.sample_name[index]
        data_numpy = np.array(data_numpy)  # nctv

        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM

        # data transform
        if self.decouple_spatial:
            data_numpy = decouple_spatial(data_numpy, edges=self.edge)
        if self.num_skip_frame is not None:
            velocity = decouple_temporal(data_numpy, self.num_skip_frame)
            C, T, V, M = velocity.shape
            data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)

        # data_numpy = pad_recurrent_fix(data_numpy, self.window_size)  # if short: pad recurrent
        # data_numpy = uniform_sample_np(data_numpy, self.window_size)  # if long: resize
        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
            # data_numpy = random_choose_simple(data_numpy, self.final_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        if self.center_choose:
            # data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)

        if self.mode == 'train':
            return data_numpy.astype(np.float32), label
        else:
            return data_numpy.astype(np.float32), label, sample_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def vis(data, edge, is_3d=True, pause=0.01, view=0.25, title=''):
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Qt5Agg')
    C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title)
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    import sys
    from os import path
    sys.path.append(
        path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-view, view, -view, view])
    if is_3d:
        ax.set_zlim3d(-view, view)
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[:2, t, v1, m]
                x2 = data[:2, t, v2, m]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[1, t, [v1, v2], m])
                    if is_3d:
                        pose[m][i].set_3d_properties(data[2, t, [v1, v2], m])
        fig.canvas.draw()
        plt.pause(pause)
    plt.close()
    plt.ioff()

