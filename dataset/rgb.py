import os
import sys
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
from dataset.video_data import *


class RGB(Dataset):
    def __init__(self, arg, mode):
        self.arg = arg
        self.mode = mode
        self.load_data()

    def load_data(self):
        with open(self.arg.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = int(self.label[index])
        sample_name = self.sample_name[index].split('.')[0]
        data_dir = os.path.join(self.arg.data_path, sample_name)
        imgs = sorted(os.listdir(data_dir))
        frames = list(map(os.path.join, [data_dir for _ in range(len(imgs))], imgs))

        if self.mode == 'train':
            data_numpy = train_video_simple(frames, self.arg.resize_shape, self.arg.final_shape, self.arg.mean, use_flip=self.arg.use_flip)
            return data_numpy.astype(np.float32), label
        else:
            data_numpy = val_video_simple(frames, self.arg.resize_shape, self.arg.final_shape, self.arg.mean)
            return data_numpy.astype(np.float32), label, sample_name

if __name__ == '__main__':
    from easydict import EasyDict as edict
    arg = edict({
        'data_path': '/home/lshi/Database/ntu_rgb/images_crop_square',
        'label_path': '/home/lshi/Database/ntu_60/xsub/val_label.pkl',
        'resize_shape': [20, 256, 256],
        'final_shape': [16, 224, 224],
        'mean': [0.5, 0.5, 0.5],
        'use_flip': [0, 0, 0]
    })
    data = RGB(arg, 'train')

    data.__getitem__(0)