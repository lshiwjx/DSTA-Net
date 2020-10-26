import cv2
from numpy import random as nprand
import random
import imutils
from dataset.rotation import *


def video_aug(video,
              brightness_delta=32,
              contrast_range=(0.5, 1.5),
              saturation_range=(0.5, 1.5),
              angle_range=(-30, 30),
              hue_delta=18):
    '''

    :param video: list of images
    :param brightness_delta:
    :param contrast_range:
    :param saturation_range:
    :param angle_range:
    :param hue_delta:
    :return:
    '''
    brightness_delta = brightness_delta
    contrast_lower, contrast_upper = contrast_range
    saturation_lower, saturation_upper = saturation_range
    angle_lower, angle_upper = angle_range
    hue_delta = hue_delta
    for index, img in enumerate(video):
        video[index] = img.astype(np.float32)

    # random brightness
    if nprand.randint(2):
        delta = nprand.uniform(-brightness_delta,
                               brightness_delta)
        for index, img in enumerate(video):
            video[index] += delta

    # random rotate
    if nprand.randint(2):
        angle = nprand.uniform(angle_lower,
                               angle_upper)
        for index, img in enumerate(video):
            video[index] = imutils.rotate(img, angle)

    # if nprand.randint(2):
    #     alpha = nprand.uniform(contrast_lower,
    #                            contrast_upper)
    #     for index, img in enumerate(video):
    #         video[index] *= alpha

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = nprand.randint(2)
    if mode == 1:
        if nprand.randint(2):
            alpha = nprand.uniform(contrast_lower,
                                   contrast_upper)
            for index, img in enumerate(video):
                video[index] *= alpha

    # convert color from BGR to HSV
    for index, img in enumerate(video):
        video[index] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # random saturation
    if nprand.randint(2):
        for index, img in enumerate(video):
            video[index][..., 1] *= nprand.uniform(saturation_lower,
                                                   saturation_upper)

    # random hue
    if nprand.randint(2):
        for index, img in enumerate(video):
            video[index][..., 0] += nprand.uniform(-hue_delta, hue_delta)
            video[index][..., 0][video[index][..., 0] > 360] -= 360
            video[index][..., 0][video[index][..., 0] < 0] += 360

    # convert color from HSV to BGR
    for index, img in enumerate(video):
        video[index] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # random contrast
    if mode == 0:
        if nprand.randint(2):
            alpha = nprand.uniform(contrast_lower,
                                   contrast_upper)
            for index, img in enumerate(video):
                video[index] *= alpha

    # randomly swap channels
    # if nprand.randint(2):
    #     for index, img in enumerate(video):
    #         video[index] = img[..., nprand.permutation(3)]

    return video


def expand_list(l, length):
    if len(l) < length:
        while len(l) < length:
            tmp = []
            [tmp.extend([x, x]) for x in l]
            l = tmp
        return sample_uniform_list(l, length)
    else:
        return l


def sample_uniform_list(l, length):
    if len(l)==length:
        return l
    interval = len(l) / length
    uniform_list = [int(i * interval) for i in range(length)]
    tmp = [l[x] for x in uniform_list]
    return tmp


def uniform_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[:, uniform_list]


def random_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data_numpy[:, random_list]


def random_choose_simple(data_numpy, size, center=False):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[0.0],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)  # 需要变换的帧的段数 0, 16, 32
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):  # 使得每一帧的旋转都不一样
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_move_whole(data_numpy, agx=0, agy=0, s=1):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.transpose((1, 2, 3, 0)).reshape(-1, C)

    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])

    data_numpy = np.dot(np.reshape(data_numpy, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
    data_numpy = data_numpy.reshape((T, V, M, C)).transpose((3, 0, 1, 2))
    return data_numpy.astype(np.float32)


def rot_to_fix_angle_fstframe(skeleton, jpts=[0, 1], axis=[0, 0, 1], frame=0, person=0):
    '''
    :param skeleton: c t v m
    :param axis: 001 for z, 100 for x, 010 for y
    '''
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    joint_bottom = skeleton[person, frame, jpts[0]]
    joint_top = skeleton[person, frame, jpts[1]]
    axis_c = np.cross(joint_top - joint_bottom, axis)
    angle = angle_between(joint_top - joint_bottom, axis)
    matrix_z = rotation_matrix(axis_c, angle)
    tmp = np.dot(np.reshape(skeleton, (-1, 3)), matrix_z.transpose())
    skeleton = np.reshape(tmp, skeleton.shape)
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_fstframe(skeleton, jpt=0, frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, frame, jpt].copy()  # c
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        mask = (person.sum(-1) != 0).reshape(T, V, 1)  # only for none zero frames
        skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_perframe(skeleton, jpt=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, :, jpt].copy().reshape((T, 1, C))  # tc
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        skeleton[i_p] = (skeleton[i_p] - main_body_center)  # TVC-T1C
    return skeleton.transpose((3, 1, 2, 0))


def decouple_spatial(skeleton, edges=()):
    tmp = np.zeros(skeleton.shape)
    for v1, v2 in edges:
        tmp[:, :, v2, :] = skeleton[:, :, v2] - skeleton[:, :, v1]
    return tmp


def obtain_angle(skeleton, edges=()):
    tmp = skeleton.copy()
    for v1, v2 in edges:
        v1 -= 1
        v2 -= 1
        x = skeleton[0, :, v1, :] - skeleton[0, :, v2, :]
        y = skeleton[1, :, v1, :] - skeleton[1, :, v2, :]
        z = skeleton[2, :, v1, :] - skeleton[2, :, v2, :]
        atan0 = np.arctan2(y, x) / 3.14
        atan1 = np.arctan2(z, x) / 3.14
        atan2 = np.arctan2(z, y) / 3.14
        t = np.stack([atan0, atan1, atan2], 0)
        tmp[:, :, v1, :] = t
    return tmp


def decouple_temporal(skeleton, inter_frame=1):  # CTVM
    skeleton = skeleton[:, ::inter_frame]
    diff = skeleton[:, 1:] - skeleton[:, :-1]
    return diff


def norm_len_fstframe(skeleton, jpts=[0, 1], frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_spine = np.linalg.norm(skeleton[person, frame, jpts[0]] - skeleton[person, frame, jpts[1]])
    if main_body_spine == 0:
        print('zero bone')
    else:
        skeleton /= main_body_spine
    return skeleton.transpose((3, 1, 2, 0))


def random_move_joint(data_numpy, sigma=0.1):  # 只随机扰动坐标点
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    rand_joint = np.random.randn(C, T, V, M) * sigma

    return data_numpy + rand_joint


def pad_recurrent(data):
    skeleton = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to  M, T, V, C
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:  # TVC 去掉头空帧，然后对齐到顶端
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            person *= 0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:  # 循环pad之前的帧
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    skeleton[i_p, i_f:] = pad
                    break
    return skeleton.transpose((3, 1, 2, 0))  # ctvm


def pad_recurrent_fix(data, length):  # CTVM
    if data.shape[1] < length:
        num = int(np.ceil(length / data.shape[1]))
        data = np.concatenate([data for _ in range(num)], 1)[:, :length]
    return data


def pad_zero(data, length):
    if data.shape[1] < length:
        new = np.zeros([data.shape[0], length - data.shape[1], data.shape[2], data.shape[3]])
        data = np.concatenate([data, new], 1)
    return data


import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import warnings

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def zoom_T(p, target_l=64):
    '''

    :param p: ctv
    :param target_l:
    :return:
    '''
    C, T, V, M = p.shape
    p_new = np.empty([C, target_l, V, M])
    for m in range(M):
        for v in range(V):
            for c in range(C):
                p_new[c, :, v, m] = inter.zoom(p[c, :, v, m], target_l / T)[:target_l]
    return p_new


def filter_T(p, kernel_size=3):
    C, T, V, M = p.shape
    p_new = np.empty([C, T, V, M])
    for m in range(M):
        for v in range(V):
            for c in range(C):
                p_new[c, :, v, m] = medfilt(p[c, :, v, m], kernel_size=kernel_size)
    return p_new


def coor_to_volume(data, size):
    '''

    :param data: CTVM
    :param size: [D, H, W]
    :return: CTDHW
    '''
    C, T, V, M = data.shape
    volume = np.zeros([V * M, T, size[0], size[1], size[2]], dtype=np.float32)
    fst_ind = np.indices([T, V, M])[0]  # T, V, M
    # one_hots = np.concatenate([np.tile(np.eye(V), [M, 1]), np.repeat(np.eye(M), V, axis=0)], axis=1).reshape(
    #     (V, M, V + M)).transpose((2, 0, 1))
    one_hots = np.eye(V * M).reshape((M, V, V * M)).transpose((2, 1, 0))  # C, V, M
    scd_inds = (data[::-1, :, :, :] * (np.array(size) - 1)[:, np.newaxis, np.newaxis, np.newaxis]).astype(
        np.long)  # 3, T, V, M
    scd_inds = np.split(scd_inds, 3, axis=0)
    volume[:, fst_ind, scd_inds[0][0], scd_inds[1][0], scd_inds[2][0]] = one_hots[:, np.newaxis, :, :]
    return volume


def coor_to_sparse(data, size, dilate_value=0, edges=None):
    '''

    :param data: CTVM
    :param size: [D, H, W]
    :return: coords->TVMx(MC)
    '''
    C, T, V, M = data.shape
    # features = np.tile(np.concatenate([np.tile(np.eye(V), [M, 1]), np.repeat(np.eye(M), V, axis=0)], axis=1).reshape(
    #     (V, M, V + M)), [T, 1, 1, 1]).reshape((T * V * M, V + M))
    features = np.tile(np.eye(V * M).reshape((M, V, V * M)).transpose((1, 0, 2)), [T, 1, 1, 1])
    coords = (data * (np.array(size) - 1)[::-1, np.newaxis, np.newaxis, np.newaxis])
    coords = np.concatenate([np.repeat(np.array(list(range(T))), V * M).reshape(1, T, V, M), coords], axis=0)
    coords = coords.transpose((1, 2, 3, 0)).astype(np.int32)

    if edges is not None:
        ecoords = []
        efeatures = []
        for t in range(T):
            for m in range(M):
                for edge in edges:
                    f1 = features[t, edge[0], m]
                    f2 = features[t, edge[1], m]
                    c1 = coords[t, edge[0], m]
                    c2 = coords[t, edge[1], m]
                    c = max(np.abs(c2 - c1))
                    ecoords.extend(
                        np.array([np.linspace(cc1, cc2, c) for cc1, cc2 in zip(c1, c2)]).transpose((1, 0)).astype(
                            np.int))
                    efeatures.extend([np.maximum(f1, f2) for _ in range(c)])
        features = np.concatenate([features.reshape((T * V * M, V * M)), efeatures], axis=0)
        coords = np.concatenate([coords.reshape((T * V * M, C + 1)), ecoords], axis=0)
    else:
        features = features.reshape((T * V * M, V * M))
        coords = coords.reshape((T * V * M, C + 1))

    coords_new = []
    features_new = []
    if dilate_value == 0:  # remove pts
        for i, coord in enumerate(coords):
            if list(coord) in coords_new:
                ind = coords_new.index(list(coord))
                features_new[ind] = np.maximum(features[i], features_new[ind])
            else:
                coords_new.append(list(coord))
                features_new.append(features[i])
    else:
        dilates = list(range(-dilate_value, dilate_value + 1))
        for i, coord in enumerate(coords):
            for j in range(C):
                for k in dilates:
                    coord_e = coord.copy()
                    coord_e[-j] += k
                    if list(coord_e) in coords_new:
                        ind = coords_new.index(list(coord_e))
                        features_new[ind] = np.maximum(features[i], features_new[ind])
                    else:
                        coords_new.append(list(coord_e))
                        features_new.append(features[i])

    return np.array(coords_new, dtype=np.int32), np.array(features_new, dtype=np.float32)


def judge_type(paths, final_shape):
    if type(paths[0]) is str:
        try:
            img = cv2.imread(paths[0])
            pre_shape = [len(paths), *img.shape]
        except:
            print(paths[0], ' is wrong')
            pre_shape = [len(paths), *final_shape[1:]]
    else:
        pre_shape = [len(paths), *paths[0].shape]

    return pre_shape


def crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug):
    imgs_crop = imgs[starts[0]:starts[0] + cshape[0]]  # TODO: paths < cshape[0]
    imgs_final = sample_uniform_list(imgs_crop, final_shape[0])

    if other_aug:
        imgs_final = video_aug(imgs_final)
    clip = []
    for index, img in enumerate(imgs_final):
        clip.append(cv2.resize(img[starts[1]:starts[1] + cshape[1], starts[2]:starts[2] + cshape[2]],
                               (final_shape[2], final_shape[1])).astype(np.float32) / 255 - mean)
    clip = np.transpose(np.array(clip, dtype=np.float32), (3, 0, 1, 2))
    for i, f in enumerate(use_flip):
        if f:
            clip = np.flip(clip, i + 1).copy()  # avoid negative strides
    return clip


def resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug):
    imgs_resize = np.array(sample_uniform_list(imgs, resize_shape[0]))
    imgs_crop = imgs_resize[starts[0]:starts[0] + final_shape[0]]
    if other_aug:
        imgs_crop = video_aug(imgs_crop)

    clip = []
    for index, img in enumerate(imgs_crop):
        clip.append(cv2.resize(img, (resize_shape[2], resize_shape[1]))[starts[1]:starts[1] + final_shape[1],
                    starts[2]:starts[2] + final_shape[2]].astype(
            np.float32) / 255 - mean)

    clip = np.transpose(np.array(clip, dtype=np.float32), (3, 0, 1, 2))
    for i, f in enumerate(use_flip):
        if f:
            clip = np.flip(clip, i + 1).copy()  # avoid negative strides
    return clip


def pose_flip(pose, use_flip):
    '''

    :param pose: T V C[x,y] M
    :param use_flip: 
    :return: 
    '''
    pose_new = pose
    if use_flip[0]:
        pose_new = pose[::-1]
    if use_flip[1] and use_flip[2]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
    elif use_flip[1]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
        pose_new[:, :, 0, :] = pose[:, :, 0, :]
    elif use_flip[2]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
        pose_new[:, :, 1, :] = pose[:, :, 1, :]

    return pose_new


def pose_crop(pose_old, start, cshape, width, height):
    '''

    :param pose_old: T V C M
    :param start: T H,W
    :param cshape:T H,W
    :param width: 
    :param height: 
    :return: 
    '''
    # temporal crop
    pose_new = pose_old[start[0]:start[0] + cshape[0]]
    T, V, C, M = pose_new.shape
    # 复原到图像大小
    pose_new = pose_new * (np.array([width, height]).reshape([1, 1, C, 1]))  # T V C M
    # 减去边框
    pose_new -= np.array([start[2], start[1]]).reshape([1, 1, C, 1])
    # 小于0的置0
    pose_new[(np.min(pose_new, -2) < 0).reshape(T, V, 1, M).repeat(C, -2)] = 0
    # 新位置除以crop后的大小
    pose_new /= np.array([cshape[2], cshape[1]]).reshape([1, 1, C, 1])
    # 大于1的值1
    pose_new[(np.max(pose_new, -2) > 1).reshape(T, V, 1, M).repeat(C, -2)] = 0
    return pose_new


def gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=False):
    try:
        if type(paths[0]) is str:
            imgs = []
            paths = sample_uniform_list(paths, resize_shape[0])
            for path in paths:
                try:
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                except:
                    print(path, ' is wrong')
                    img = np.zeros([*final_shape[1:], 3], dtype=np.uint8)
                imgs.append(img)
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug)
            return clip
        elif type(paths[0]) is tuple:
            imgs, poses = np.array([i[0] for i in paths]), np.array([i[1] for i in paths])
            if len(imgs) != len(poses):
                imgs = np.array(sample_uniform_list(imgs, len(poses)))
            if poses.shape[2] >= 3:  # T,V,C,M
                poses = poses[:, :, :2]
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug).transpose(
                (1, 0, 2, 3))
            poses = np.array(sample_uniform_list(poses, resize_shape[0]))
            poses = pose_crop(poses, starts, final_shape, resize_shape[2], resize_shape[1])
            poses = pose_flip(poses, use_flip)
            return clip, poses
        else:
            imgs = paths
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug)
            return clip
    except:
        print(paths)


def gen_clip(paths, starts, cshape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    try:
        if type(paths[0]) is str:
            imgs = []
            for path in paths:
                try:
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                except:
                    print(path, ' is wrong')
                    img = np.zeros([*final_shape[1:], 3], dtype=np.uint8)
                imgs.append(img)
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug)
            return clip
        elif type(paths[0]) is tuple:
            imgs, poses = np.array([i[0] for i in paths]), np.array([i[1] for i in paths])
            if len(imgs) != len(poses):
                imgs = np.array(sample_uniform_list(imgs, len(poses)))
            if poses.shape[2] >= 3:  # T,V,C,M
                poses = poses[:, :, :2]
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug).transpose(
                (1, 0, 2, 3)).copy()
            poses = poses[starts[0]:starts[0] + cshape[0]]
            poses = pose_crop(poses, starts, cshape, imgs[0].shape[1], imgs[0].shape[0])
            poses = pose_flip(poses, use_flip).copy()
            return clip, poses
        else:
            imgs = paths
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug)
            return clip
    except:
        print(paths)


def train_video_simple(paths, resize_shape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    """

    :param paths: [frame1, frame2 ....] 
    :param resize_shape:  [l, h, w] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [0,0,0]
    :return: 
    """
    gap = [resize_shape[i] - final_shape[i] for i in range(3)]

    starts = [int(a * random.random()) for a in gap]

    clip = gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=other_aug)

    return clip


def val_video_simple(paths, resize_shape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    """

    :param paths: [frame1, frame2 ....] 
    :param resize_shape:  [l, h, w] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [0,0,0
    :return: 
    """

    gap = [resize_shape[i] - final_shape[i] for i in range(3)]

    starts = [int(a * 0.5) for a in gap]
    clip = gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=other_aug)

    return clip


def eval_video(paths, crop_ratios, crop_positions, final_shape, mean, use_flip=(0, 0, 0)):
    """

    :param paths: [frame1, frame2 ....] 
    :param crop_ratios:  [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param crop_positions: [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [False, False, False]
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    clips = []
    for crop_t in crop_ratios[0]:
        for crop_h in crop_ratios[1]:
            for crop_w in crop_ratios[2]:
                cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

                gap = [pre_shape[i] - cshape[i] for i in range(3)]
                for p_t in crop_positions[0]:
                    for p_h in crop_positions[1]:
                        for p_w in crop_positions[2]:
                            starts = [int(a * b) for a in gap for b in [p_t, p_h, p_w]]
                            clip = gen_clip(paths, starts, cshape, final_shape, mean)
                            clips.append(clip)  # clhw
                            for i, f in enumerate(use_flip):
                                if f:
                                    clip_flip = np.flip(clip, i + 1).copy()
                                    clips.append(clip_flip)

    return clips


def train_video(paths, crop_ratios, crop_positions, final_shape, mean, use_flip=(0, 0, 0)):
    """

    :param paths: [frame1, frame2 ....] 
    :param crop_ratios:  [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param crop_positions: [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: True or False
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    crop_t = random.sample(crop_ratios[0], 1)[0]
    crop_h = random.sample(crop_ratios[1], 1)[0]
    crop_w = random.sample(crop_ratios[2], 1)[0]
    cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

    gap = [pre_shape[i] - cshape[i] for i in range(3)]

    p_t = random.sample(crop_positions[0], 1)[0]
    p_h = random.sample(crop_positions[1], 1)[0]
    p_w = random.sample(crop_positions[2], 1)[0]

    starts = [int(a * b) for a, b in list(zip(gap, [p_t, p_h, p_w]))]
    clip = gen_clip(paths, starts, cshape, final_shape, mean, use_flip, other_aug=True)

    # for i, f in enumerate(use_flip):
    #     if f:
    #         clip = np.flip(clip, i + 1)

    return clip


def val_video(paths, final_shape, mean):
    """

    :param paths: [frame1, frame2 ....] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    crop_t = 1
    crop_h = 1
    crop_w = 1
    cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

    gap = [pre_shape[i] - cshape[i] for i in range(3)]

    p_t = 0.5
    p_h = 0.5
    p_w = 0.5

    starts = [int(a * b) for a, b in list(zip(gap, [p_t, p_h, p_w]))]
    clip = gen_clip(paths, starts, cshape, final_shape, mean)

    return clip
