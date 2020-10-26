import pickle
from tqdm import tqdm
import sys
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
        coords_frame = np.array(nums[1:]).reshape((21, 3)).astype(np.float32)
        skeletons.append(coords_frame)
    num_frame = len(skeletons)
    skeletons = np.expand_dims(np.array(skeletons).transpose((2, 0, 1)), axis=-1)  # CTVM
    skeletons = np.transpose(skeletons, [3, 1, 2, 0])  # M, T, V, C
    return skeletons, num_frame


def gendata():
    root = '/home/lshi/Database/fpha_hand/'
    split_txt = 'data_split_action_recognition.txt'
    ignores = ['Subject_2/close/milk/4', 'Subject_2/put_tea_bag/2', 'Subject_4/filp_sponge/2',
               'Subject_2/open_letter/2', 'Subject_2/open_peanut_butter/1']
    ske_root = os.path.join(root, 'Hand_pose_annotation_v1')
    split = open(os.path.join(root, split_txt), 'r').readlines()
    train_items = split[1:1 + 600]
    val_items = split[2 + 600:]
    sets = {
        'val': val_items,
        'train': train_items,
    }
    for sub, items in sets.items():
        num_frames = []
        skeletons_all = []
        names_all = []
        labels_all = []
        for item in tqdm(items):
            path = item.split(' ')[0]
            if path in ignores:
                continue
            ske_txt = os.path.join(ske_root, path, 'skeleton.txt')
            skeletons, num_frame = read_skeleton(ske_txt)  # CTVM
            skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 3], zaxis=[0, 3], xaxis=[3, 2])
            # ske_vis(skeletons, view=1, pause=0.1)
            label = int(item.split(' ')[1][:-1])
            labels_all.append(label)
            names_all.append(path)
            skeletons_all.append(skeletons)  # NCTVM
            num_frames.append(num_frame)
        print(max(num_frames))  # 7 11 ...  371, 420, 1151   val: 8-341, 451, 830
        pickle.dump(skeletons_all, open(os.path.join(root, '{}_skeleton.pkl'.format(sub)), 'wb'))
        pickle.dump([names_all, labels_all], open(os.path.join(root, '{}_label.pkl'.format(sub)), 'wb'))


def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.fpha_skeleton import edge
    vis(data, edge=edge, **kwargs)


if __name__ == '__main__':
    gendata()
    # test_wrong_data()
    # test()
