import pickle
from tqdm import tqdm
import sys
from dataset.rotation import *
from dataset.normalize_skeletons import normalize_skeletons

sys.path.extend(['../../'])

import numpy as np
import os



def read_skeleton(ske_txt):
    ske_txt = open(ske_txt, 'r').readlines()
    skeletons = []
    for line in ske_txt:
        nums = line.split(' ')
        # num_frame = int(nums[0]) + 1
        coords_frame = np.array(nums).reshape((22, 3)).astype(np.float32)
        skeletons.append(coords_frame)
    num_frame = len(skeletons)
    skeletons = np.expand_dims(np.array(skeletons).transpose((2, 0, 1)), axis=-1)  # CTVM
    skeletons = np.transpose(skeletons, [3, 1, 2, 0])  # M, T, V, C
    return skeletons, num_frame


def gendata():
    root = '/your/path/to/shrec_hand/'
    train_split = open(os.path.join(root, 'train_gestures.txt'), 'r').readlines()
    val_split = open(os.path.join(root, 'test_gestures.txt'), 'r').readlines()

    skeletons_all_train = []
    names_all_train = []
    labels14_all_train = []
    labels28_all_train = []
    skeletons_all_val = []
    names_all_val = []
    labels14_all_val = []
    labels28_all_val = []

    for line in tqdm(train_split):
        line = line.rstrip()
        g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])
        # ske_vis(skeletons, view=1, pause=0.1)
        skeletons_all_train.append(skeletons)
        labels14_all_train.append(label_14-1)
        labels28_all_train.append(label_28-1)
        names_all_train.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))

    pickle.dump(skeletons_all_train, open(os.path.join(root, 'train_skeleton.pkl'), 'wb'))
    pickle.dump([names_all_train, labels14_all_train],
                open(os.path.join(root, 'train_label_14.pkl'), 'wb'))
    pickle.dump([names_all_train, labels28_all_train],
                open(os.path.join(root, 'train_label_28.pkl'), 'wb'))

    for line in tqdm(val_split):
        line = line.rstrip()
        g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])

        skeletons_all_val.append(skeletons)
        labels14_all_val.append(label_14-1)
        labels28_all_val.append(label_28-1)
        names_all_val.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))

    pickle.dump(skeletons_all_val, open(os.path.join(root, 'val_skeleton.pkl'), 'wb'))
    pickle.dump([names_all_val, labels14_all_val],
                open(os.path.join(root, 'val_label_14.pkl'), 'wb'))
    pickle.dump([names_all_val, labels28_all_val],
                open(os.path.join(root, 'val_label_28.pkl'), 'wb'))


def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.fpha_skeleton import edge
    vis(data, edge=edge, **kwargs)


if __name__ == '__main__':
    gendata()
