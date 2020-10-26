from prepare.ntu_60.gendata import *
sys.path.extend(['../../'])

import numpy as np
import os

valid_actions = [5, 6, 7, 8, 9]
valid_actions15 = [5, 6, 7, 8, 9, 10, 22, 23, 24, 25, 26, 27, 31, 32, 43]


def gendata_for_transfer(data_path, out_path, ignored_sample_path=None, benchmark='transfer', part='eval', training_subjects=None):
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

        if action_class in valid_actions:
            action_class = valid_actions.index(action_class)
        else:
            continue

        istraining = (camera_id in training_cameras) and (subject_id in training_subjects) and (setup_class % 2 == 1)
        isval = (camera_id not in training_cameras) and (subject_id not in training_subjects) and (setup_class % 2 == 0)

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = isval
        else:
            raise ValueError()

        if issample:
            sample_names.append(filename)
            sample_labels.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:  # 1326 train 3382 val 290
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/home/lshi/Database/ntu_60_raw/nturgb+d_skeletons')
    parser.add_argument('--ignored_sample_path',
                        default='/home/lshi/Database/ntu_60_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/home/lshi/Database/ntu_60/')

    benchmark = ['transfer']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata_for_transfer(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                training_subjects=training_subjects)
