def view_raw_skeletons_and_images():
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt

    ignored_sample_path = '/home/lshi/Database/ntu_60_raw/samples_with_missing_skeletons.txt'
    raw_root = '/home/lshi/Database/ntu_60_raw/nturgb+d_skeletons'
    rgb_root = '/home/lshi/Database/ntu_rgb/images_full/'
    for i, filename in enumerate(sorted(os.listdir(raw_root))[::-1]):
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
        # if not i%3==0:
        #     continue
        if filename in ignored_samples:
            continue
        print(i)
        ske = read_skeleton_filter(os.path.join(raw_root, filename))
        bdys = get_body_info(ske)
        if len(bdys) > 2 or len(bdys) == 0:
            print(filename)
        bdys = filter_body(bdys)
        skeletons = normalize_skeletons(bdys.copy())  # ctvm
        ske_vis(skeletons)

        positions = bdys[:, -2:]  # mctv
        positions = positions[:, :, positions.sum(0).sum(0).sum(-1) != 0]
        plt.ion()
        fig = plt.figure()
        for i in range(0, positions.shape[-2], 2):
            fig.clear()
            img = os.path.join(rgb_root, filename[:-9], '{}.jpg'.format(str(i + 1).zfill(5)))
            plt.imshow(plt.imread(img))
            for body in bdys:
                x, y = body[-2:, i]
                plt.scatter(x, y, s=2)
            fig.canvas.draw()
            plt.pause(0.01)
        plt.close()
        plt.ioff()
    print('finish')


def view_skeletons_on_cropped_images():
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt

    rgb_root = '/home/lshi/Database/ntu_rgb/images_full/'
    position_path = '/home/lshi/Database/ntu_lshipre_120/xsub/val_joint_position_in_img.npy'
    label_path = '/home/lshi/Database/ntu_lshipre_120/xsub/val_label.pkl'
    data_path = '/home/lshi/Database/ntu_lshipre_120/xsub/val_data_joint.npy'
    files, labels = pickle.load(open(label_path, 'rb'))
    positions = np.load(position_path)
    data = np.load(data_path, mmap_mode='r')  # nctvm
    for i, filename in enumerate(files):
        print(i)
        skeletons = data[i]
        ske_vis(skeletons)

        bdys = positions[i].transpose((3, 0, 1, 2))  # ctvm-mctv
        bdys = bdys[:, -2:]  # mctv
        bdys = bdys[:, :, bdys.sum(0).sum(0).sum(-1) != 0]
        plt.ion()
        fig = plt.figure()
        for i in range(0, bdys.shape[-2], 2):
            fig.clear()
            img = os.path.join(rgb_root, filename[:-9], '{}.jpg'.format(str(i + 1).zfill(5)))
            plt.imshow(plt.imread(img))
            for body in bdys:
                x, y = body[-2:, i]
                plt.scatter(x, y, s=2)
            fig.canvas.draw()
            plt.pause(0.01)
        plt.close()
        plt.ioff()
    print('finish')