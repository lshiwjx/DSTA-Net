def view_wrong_classified_skeletons():
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt
    wrong_list = '/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/agcnv2/val/ntucqcaosmall_agcn_xsub_30032f_drop15_norand_drop15_304050_sgdnev_g4/wrong_path_pre_true.txt'
    label_file = open("/home/lshi/Project/Pytorch/st-gcn/tools/ntu_label_simple.txt")
    classes = label_file.readlines()
    classes = [x[:-1] for x in classes]
    lines = open(wrong_list).readlines()

    rgb_root = '/home/lshi/Database/ntu_rgb/images_full/'
    position_path = '/home/lshi/Database/ntu_lshi_pre/xsub/val_joint_position_in_img.npy'
    label_path = '/home/lshi/Database/ntu_lshi_pre/xsub/val_label.pkl'
    data_path = '/home/lshi/Database/ntu_lshi_pre/xsub/val_data_joint.npy'
    files, labels = pickle.load(open(label_path, 'rb'))
    positions = np.load(position_path)
    data = np.load(data_path)  # nctvm
    for line in lines:
        (path, pre, true) = line[:-1].split(',')
        i = files.index(path[:-4] + '.skeleton')
        print('index: ', i, 'predicted label: ', classes[int(pre)], 'true albel: ', classes[int(true)])
        skeletons = data[i]  # ctvm
        ske_vis(skeletons)

        bdys = positions[i].transpose((3, 0, 1, 2))  # ctvm-mctv
        bdys = bdys[:, -2:]  # mctv
        bdys = bdys[:, :, bdys.sum(0).sum(0).sum(-1) != 0]
        plt.ion()
        fig = plt.figure()
        for i in range(0, bdys.shape[-2], 2):
            fig.clear()
            img = os.path.join(rgb_root, path[:-4], '{}.jpg'.format(str(i + 1).zfill(5)))
            plt.imshow(plt.imread(img))
            for body in bdys:
                x, y = body[-2:, i]
                plt.scatter(x, y, s=2)
            fig.canvas.draw()
            plt.pause(0.01)
        plt.close()
        plt.ioff()
    print('finish')

def test_wrong(data_path, label_path, wrong_list=None, edge=None, is_3d=False, mode='train'):
    import glob
    loader = DataLoader(
        dataset=DHG_SKE(data_path, label_path, window_size=120, final_size=100, random_choose=False, mode=mode),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    root = '/home/lshi/Database/dhg_hand_shrec/HandGestureDataset_SHREC2017'
    label_file = open("/home/lshi/Project/Pytorch/3d-resnet-pytorch/prepare/dha_hand/dha_label.txt")
    data_file = root + '/test_gestures.txt'
    paths = open(data_file).readlines()
    classes = label_file.readlines()
    classes = [x[:-1] for x in classes]
    lines = open(wrong_list).readlines()
    sample_name = loader.dataset.sample_name

    for line in lines:
        (path, pre, true) = line[:-1].split(',')
        path = int(path[7:-1])
        print(path, classes[int(pre)], classes[int(true)])

        index = sample_name.index(path)
        if mode != 'train':
            data, label, index = loader.dataset[index]
        else:
            data, label = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        rgb_path = paths[path].split()
        joint_path = root + "/gesture_{}/finger_{}/subject_{}/essai_{}/".format(rgb_path[0], rgb_path[1], rgb_path[2],
                                                                                rgb_path[3])
        plt.ion()
        fig = plt.figure()
        for rgb in sorted(glob.glob(joint_path + '*.png'), key=lambda x: int(x.split('/')[-1][:-10])):
            fig.clear()
            img = plt.imread(rgb)
            plt.imshow(img)
            fig.canvas.draw()
            plt.pause(0.1)
        plt.close()
        plt.ioff()

        vis(data, is_3d, edge, pause=0.1, view=1)
