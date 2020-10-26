import argparse
import pickle
import os
import sys
from prepare.ntu_60.gendata import gendata

sys.path.extend(['../../'])

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                     38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                     80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
max_channel = 5  # xyz+xy
num_channel = 3
channel_name = ['x', 'y', 'z', 'colorX', 'colorY']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/home/lshi/Database/ntu_120_raw/')
    parser.add_argument('--ignored_sample_path',
                        default='/home/lshi/Database/ntu_120_raw/samples_with_missing_skeletons_120.txt')
    parser.add_argument('--out_folder', default='/home/lshi/Database/ntu_120/')

    benchmark = ['xset', 'xsub']
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
                part=p, training_subjects=training_subjects)
