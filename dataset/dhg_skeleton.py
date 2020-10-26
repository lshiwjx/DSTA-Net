
from torch.utils.data import DataLoader, Dataset
from dataset.skeleton import Skeleton, vis


edge = ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21))


class DHG_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False):
        super().__init__(data_path, label_path, window_size, final_size, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose)
        self.edge = edge


def test(data_path, label_path, vid=None, edge=None, is_3d=False, mode='train'):
    loader = DataLoader(
        dataset=DHG_SKE(data_path, label_path, window_size=150, final_size=128, mode=mode,
                        random_choose=True, center_choose=False, decouple_spatial=False, num_skip_frame=None),
        batch_size=1,
        shuffle=False,
        num_workers=0)

    labels = open('../prepare/shrec/label_28.txt', 'r').readlines()
    for i, (data, label) in enumerate(loader):
        if i%100==0:
            vis(data[0].numpy(), edge=edge, view=0.2, pause=0.01, title=labels[label.item()].rstrip())

    sample_name = loader.dataset.sample_name
    index = sample_name.index(vid)
    if mode != 'train':
        data, label, index = loader.dataset[index]
    else:
        data, label = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=0.2, pause=0.1, title=labels[label].rstrip())


if __name__ == '__main__':
    data_path = "/your/path/to/shrec_hand/train_skeleton.pkl"
    label_path = "/your/path/to/shrec_hand/train_label_28.pkl"
    # data_path = "/your/path/to/dhg_hand_shrec/train_skeleton_ddnet.pkl"
    # label_path = "/your/path/to/dhg_hand_shrec/train_label_ddnet_14.pkl"
    # test(data_path, label_path, vid=1, edge=edge, is_3d=True, mode='train')
    test(data_path, label_path, vid='14_2_27_5', edge=edge, is_3d=True, mode='train')
