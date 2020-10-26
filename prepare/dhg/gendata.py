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
    skeletons = np.transpose(skeletons, [3, 1, 2, 0])  # CTVM-MTVC
    return skeletons, num_frame


def gendata():
    root = '/your/path/to/dhg_hand/'
    split_txt = 'informations_troncage_sequences.txt'
    split = open(os.path.join(root, split_txt), 'r').readlines()
    num_sub = 20
    skeletons_all_train = [[] for i in range(num_sub)]
    names_all_train = [[] for i in range(num_sub)]
    labels14_all_train = [[] for i in range(num_sub)]
    labels28_all_train = [[] for i in range(num_sub)]
    skeletons_all_val = [[] for i in range(num_sub)]
    names_all_val = [[] for i in range(num_sub)]
    labels14_all_val = [[] for i in range(num_sub)]
    labels28_all_val = [[] for i in range(num_sub)]
    for line in tqdm(split):
        line = line.split("\n")[0]
        data = line.split(" ")
        g_id = data[0]
        f_id = data[1]
        sub_id = data[2]
        e_id = data[3]
        start_frame = int(data[4])
        end_frame = int(data[5])
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        skeletons = skeletons[:, start_frame:end_frame + 1]
        skeletons = normalize_skeletons(skeletons, origin=0)
        # ske_vis(skeletons, view=1, pause=0.1)
        label14 = int(g_id) - 1
        if int(f_id) == 1:
            label28 = int(g_id) - 1
        else:
            label28 = int(g_id) - 1 + 14
        for id in range(num_sub):
            if id == int(sub_id) - 1:
                skeletons_all_val[id].append(skeletons)
                labels14_all_val[id].append(label14)
                labels28_all_val[id].append(label28)
                names_all_val[id].append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))
            else:
                skeletons_all_train[id].append(skeletons)
                labels14_all_train[id].append(label14)
                labels28_all_train[id].append(label28)
                names_all_train[id].append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))
    for id in range(num_sub):
        pickle.dump(skeletons_all_train[id], open(os.path.join(root, 'train_skeleton_{}.pkl'.format(id)), 'wb'))
        pickle.dump(skeletons_all_val[id], open(os.path.join(root, 'val_skeleton_{}.pkl'.format(id)), 'wb'))
        pickle.dump([names_all_train[id], labels14_all_train[id]],
                    open(os.path.join(root, 'train_label_{}_14.pkl'.format(id)), 'wb'))
        pickle.dump([names_all_val[id], labels14_all_val[id]],
                    open(os.path.join(root, 'val_label_{}_14.pkl'.format(id)), 'wb'))
        pickle.dump([names_all_train[id], labels28_all_train[id]],
                    open(os.path.join(root, 'train_label_{}_28.pkl'.format(id)), 'wb'))
        pickle.dump([names_all_val[id], labels28_all_val[id]],
                    open(os.path.join(root, 'val_label_{}_28.pkl'.format(id)), 'wb'))


def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.dhg_skeleton import edge
    vis(data, edge=edge, **kwargs)


if __name__ == '__main__':
    gendata()