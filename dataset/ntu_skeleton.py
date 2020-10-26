
import pickle
from torch.utils.data import DataLoader, Dataset
from dataset.video_data import *
from dataset.skeleton import Skeleton, vis

edge = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))


class NTU_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False):
        super().__init__(data_path, label_path, window_size, final_size, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose)
        self.edge = edge

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path, mmap_mode='r')[:, :3]  # NCTVM


def test(data_path, label_path, vid=None, edge=None, is_3d=False, mode='train'):
    dataset = NTU_SKE(data_path, label_path, window_size=48, final_size=32, mode=mode,
                      random_choose=True, center_choose=False)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    labels = open('../prepare/ntu_120/label.txt', 'r').readlines()
    for i, (data, label) in enumerate(loader):
        if i%1000==0:
            vis(data[0].numpy(), edge=edge, view=1, pause=0.01, title=labels[label.item()].rstrip())

    sample_name = loader.dataset.sample_name
    sample_id = [name.split('.')[0] for name in sample_name]
    index = sample_id.index(vid)
    if mode != 'train':
        data, label, index = loader.dataset[index]
    else:
        data, label = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=1, pause=0.1)


if __name__ == '__main__':
    data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
    # data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    # label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    # test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
