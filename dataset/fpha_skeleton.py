from torch.utils.data import DataLoader, Dataset
from dataset.skeleton import Skeleton, vis


edge = ((0, 1), (1, 6), (6, 11), (11, 16),
        (0, 2), (2, 7), (7, 12), (12, 17),
        (0, 3), (3, 8), (8, 13), (13, 18),
        (0, 4), (4, 9), (9, 14), (14, 19),
        (0, 5), (5, 10), (10, 15), (15, 20))


class FPHA_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False):
        super().__init__(data_path, label_path, window_size, final_size, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose)
        self.edge = edge


def test(data_path, label_path, vid=None, edge=None, is_3d=False, mode='train'):
    loader = DataLoader(
        dataset=FPHA_SKE(data_path, label_path, window_size=120, final_size=16, mode=mode),
        batch_size=1,
        shuffle=False,
        num_workers=0)

    sample_name = loader.dataset.sample_name
    index = sample_name.index(vid)
    if mode != 'train':
        data, label, index = loader.dataset[index]
    else:
        data, label = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=1, pause=0.1)


if __name__ == '__main__':
    data_path = "/home/lshi/Database/fpha_hand/val_skeleton.pkl"
    label_path = "/home/lshi/Database/fpha_hand/val_label.pkl"
    test(data_path, label_path, vid='Subject_1/open_juice_bottle/1', edge=edge, is_3d=True, mode='train')
