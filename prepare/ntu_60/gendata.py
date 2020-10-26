import argparse
import pickle
from tqdm import tqdm
import sys
from dataset.normalize_skeletons import normalize_skeletons

sys.path.extend(['../../'])
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
max_channel = 5  # xyz+xy
num_channel = 3
channel_name = ['x', 'y', 'z', 'colorX', 'colorY']

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['bodyID'] = int(body_info['bodyID'])
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_body_info(skeletons):
    num_frames = skeletons['numFrame']
    bodys = {}
    for index_f, frames in enumerate(skeletons['frameInfo']):
        num_body = frames['numBody']
        frame_bodyid = []
        for body in frames['bodyInfo']:
            body_id = body['bodyID']
            if body_id not in frame_bodyid:
                frame_bodyid.append(body_id)
            else:
                while body_id in frame_bodyid:
                    body_id += 1
            if body_id not in bodys.keys():
                bodys[body_id] = np.zeros((max_channel, max_frame, num_joint))
            for c, c_str in enumerate(channel_name):
                for j in range(num_joint):
                    bodys[body_id][c, index_f, j] = body['jointInfo'][j][c_str]
    return bodys


def get_nonzero_std(s):  # ctv
    s = s - s[:, :, 0:1]  # sub center joint
    index = s[:3].sum(0).sum(-1) != 0  # select valid frames
    s = s[:, index]
    if len(s) != 0:
        s = s[0].std() + s[1].std() + s[2].std()  # std of three channels
    else:
        s = 0
    return s


def xy_valid(body):
    '''
    Judge whether the body is valid
    :param body: 
    :return: True or False
    '''
    index = body[:num_channel].sum(0).sum(-1) != 0  # select valid frames
    body = body[:, index]
    x = body[0, 0].max() - body[0, 0].min()
    y = body[1, 0].max() - body[1, 0].min()
    return y * 0.8 > x


def filter_body(bodys):
    '''
    Filter bodys to max number person, return mctv
    :param bodys: MCTV m=5 
    :return: MCTV
    '''
    if len(bodys) == 1:
        bodys = np.array([item for k, item in bodys.items()])
        bodys = np.transpose(bodys, [0, 2, 3, 1])  # M, T, V, C
        return bodys

    bodys = np.array([item for k, item in bodys.items()])
    # bodys[:, :, :1] = 0  # remove first frame
    # bodys = bodys[bodys[:, :num_channel].sum(-1).sum(-1).sum(-1) != 0]  # remove 0 body

    # body sort by energy
    energy = np.array([get_nonzero_std(x) for x in bodys])
    index = energy.argsort()[::-1]
    bodys = bodys[index]  # 0.63 0.5

    # filter objs
    energy = np.array([get_nonzero_std(x) for x in bodys])
    energy_min = max(energy) * 0.85
    del_list = np.where(np.array(energy < energy_min) == True)[0]
    for i in del_list[::-1]:
        if not xy_valid(bodys[i]):  # delete obj should be obj
            bodys = np.concatenate([bodys[:i], bodys[i + 1:]], 0)

    # concat by durs
    # body_durs = []
    # for i, body in enumerate(bodys):
    #     valid_frames = np.where(body.sum(0).sum(-1) != 0)[0]
    #     body_durs.append([valid_frames.min(), valid_frames.max()])
    #
    # del_list = []
    # for i, (begin, end) in enumerate(body_durs):
    #     if begin == end:
    #         continue
    #     if i in del_list:
    #         continue
    #     for j, (begin2, end2) in enumerate(body_durs):
    #         if j in del_list:
    #             continue
    #         if np.abs(begin2 - end) < 10:
    #             pass
    #         if end == begin2 - 1:
    #             bodys[i] = bodys[i] + bodys[j]
    #             del_list.append(j)
    #             break
    # for i in del_list[::-1]:
    #     bodys = np.concatenate([bodys[:i], bodys[i+1:]], 0)

    # del bodys that are too short
    # body_frames = []
    # del_list = []
    # for i, body in enumerate(bodys):
    #     valid_frames = np.where(body.sum(0).sum(-1) != 0)[0]
    #     body_frames.append(valid_frames.max() - valid_frames.min())
    # body_frame_max = max(body_frames)
    # for i, f in enumerate(body_frames):
    #     if f < body_frame_max * 0.2:
    #         del_list.append(i)
    # for i in del_list[::-1]:
    #     bodys = np.concatenate([bodys[:i], bodys[i + 1:]], 0)

    # remove incomplete frames  有些双人某个人很少
    # begins = []
    # ends = []
    # for i, body in enumerate(bodys):
    #     valid_frames = np.where(body.sum(0).sum(-1) != 0)[0]
    #     begins.append(valid_frames.min())
    #     ends.append(valid_frames.max())
    # bodys[:, :, min(begins):max(begins)] = 0
    # bodys[:, :, min(ends):max(ends)] = 0

    # save max num bodys for new bodys
    energy = np.array([get_nonzero_std(x) for x in bodys])
    index = energy.argsort()[::-1][0:max_body_true]
    bodys = bodys[index]

    bodys = np.transpose(bodys, [0, 2, 3, 1])  # M, T, V, C

    return bodys


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval', training_subjects=None):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_names = []
    sample_labels = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        setup_class = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xset':
            istraining = (setup_class % 2 == 1)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_names.append(filename)
            sample_labels.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, list(sample_labels)), f)

    data_skeleton = np.zeros((len(sample_labels), num_channel, max_frame, num_joint, max_body_true), dtype=np.float32)
    data_rgb_position = np.zeros((len(sample_labels), max_channel - num_channel, max_frame, num_joint, max_body_true),
                                 dtype=np.float32)
    num_frames = []
    for i, sample_name in enumerate(tqdm(sample_names)):
        seq_info = read_skeleton_filter(os.path.join(data_path, sample_name))
        num_frames.append(seq_info['numFrame'])
        bodys = get_body_info(seq_info)
        bodys = filter_body(bodys)  # mtvc
        num_body = bodys.shape[0]
        skeletons = normalize_skeletons(bodys[..., :3], origin=0, base_bone=[0, 20], zaxis=[0, 20], xaxis=[20, 5])
        # use this to see the preprocessed skeletons
        # ske_vis(skeletons, view=1, pause=0.1)
        data_skeleton[i, :, :, :, :num_body] = skeletons  # ctvm
        data_rgb_position[i, :, :, :, :num_body] = bodys.transpose((3, 1, 2, 0))[3:5]
    print(max(num_frames))  # 15-300 300挺多的 平均75帧
    np.save('{}/{}_data_joint.npy'.format(out_path, part), data_skeleton)
    np.save('{}/{}_joint_position_in_img.npy'.format(out_path, part), data_rgb_position)


def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.ntu_skeleton import edge
    vis(data, edge=edge, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/home/lshi/Database/ntu_60_raw/nturgb+d_skeletons')
    parser.add_argument('--ignored_sample_path',
                        default='/home/lshi/Database/ntu_60_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/home/lshi/Database/ntu_60/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                training_subjects=training_subjects)
