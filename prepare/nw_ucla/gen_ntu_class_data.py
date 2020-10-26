import numpy as np
import argparse
import os
import sys
from numpy.lib.format import open_memmap
import pickle
from tqdm import tqdm
from dataset.normalize_skeletons import normalize_skeletons

training_subjects = [
    1, 2, 4, 5, 6, 7, 8, 9
]
# “pick up with one hand”, “pick up with two hands”, “drop trash”, “walk around”, “sit down”,
# “stand up”, “donning”, “doffing”, “throw” and “carry”.
# 没有对应的人的位置
# class_map = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [8, 6], [9, 7], [11, 8], [12, 9]]
class_map = [-1, 0, 1, 2, 3, 4, 5, -1, 6, 7, -1, 8, 9]
training_cameras = [1, 2]
max_body = 1
num_joint = 20
max_frame = 200


def gendata(data_path, out_path, part='val', bench='xview'):
    sample_name = []
    sample_label = []
    rgb_path = []
    cameras = [1, 2, 3]

    for cid in cameras:
        path = os.path.join(data_path, 'view_' + str(cid))
        for filename in sorted(os.listdir(path)):
            action_class = int(filename[filename.find('a') + 1:filename.find('a') + 3])
            subject_id = int(filename[filename.find('s') + 1:filename.find('s') + 3])
            envir_id = int(filename[filename.find('e') + 1:filename.find('e') + 3])
            camera_id = cid

            if bench == 'xview':
                train_val = 'train' if (camera_id in [1, 2]) else 'val'  # 123
            elif bench == 'xsub':
                train_val = 'train' if (subject_id in [1, 2, 3, 4, 5, 6, 7]) else 'val'  # 12345678910
            elif bench == 'xenv':
                train_val = 'train' if (envir_id in [0, 1, 2, 3]) else 'val'  # 01234
            else:
                raise RuntimeError()

            if train_val == part:
                sample_name.append('view_' + str(cid) + '/' + filename)
                sample_label.append(class_map[action_class])
                rgb_path.append(os.path.join(data_path, 'view_' + str(cid) + '/' + filename))

    # with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)
    # with open('{}/{}_rgb_label.pkl'.format(out_path, part), 'wb') as f:
    #     pickle.dump(rgb_path, f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body), dtype=np.float32)
    max_f = 0
    null_ske = 0
    for i, s in enumerate(tqdm(sample_name)):
        frames = []
        with open(os.path.join(data_path, s, 'fileList.txt')) as f:
            lines = f.readlines()
            for line in lines:
                item = [int(tmp) for tmp in line.rstrip().split(' ')]
                if not len(item) == 3:
                    print('frame has no skeleton data')
                    continue
                frames.append(item)
        frames.sort()
        len_t = len(frames)
        if len_t > max_f:
            max_f = len_t
        # print(len_t)
        # raise ValueError
        for j, (i_f, i_rgb, i_ske) in enumerate(frames):
            file_path = os.path.join(data_path, s, 'frame_' + str(i_f) + '_tc_' + str(i_ske) + '_skeletons.txt')
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                _ = f.readline()
                if _ is not '':
                    for k in range(num_joint):
                        try:
                            line = f.readline().rstrip().split(',')
                            x, y, z, _ = [float(tmp) for tmp in line]
                        except:
                            print('wrong')
                            continue
                        fp[i, 0, j, k, :] = x
                        fp[i, 1, j, k, :] = y
                        fp[i, 2, j, k, :] = z
                else:
                    null_ske += 1
            rgb_path = os.path.join(data_path, s, 'frame_' + str(i_f) + '_tc_' + str(i_rgb) + '_rgb.jpg')
            jpt = fp[i, :, j, :, 0]
            img = plt.imread(rgb_path)
            plt.imshow(img)
            plt.scatter(jpt[0]/jpt[2] * 200 + 330, (-jpt[1]/jpt[2] * 200 + 230))
            plt.show()
    print(max_f)
    print(null_ske)
    fp = normalize_skeletons(fp, is_3d=True, zaxis=[0, 1], xaxis=[8, 4])
    np.save('{}/{}_data.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/lshi/Database/NW-UCLA/multiview_action/')
    parser.add_argument('--out_folder', default='/home/lshi/Database/NW-UCLA/ucla/')

    benchmark = ['xview', 'xsub', 'xenv']
    part = ['val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(arg.data_path, out_path, part=p)
