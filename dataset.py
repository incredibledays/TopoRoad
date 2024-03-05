import os
import glob
import numpy as np
import cv2
import pickle
import torch
import torch.utils.data as data


def random_hue_saturation_value(sat, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(sat)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        sat = cv2.merge((h, s, v))
        sat = cv2.cvtColor(sat, cv2.COLOR_HSV2BGR)
    return sat


def random_shift_scale_rotate(sat, svx, ori, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), border_mode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = sat.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        sat = cv2.warpPerspective(sat, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        svx = cv2.warpPerspective(svx, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        ori = cv2.warpPerspective(ori, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
    return sat, svx, ori


def random_horizontal_flip(sat, svx, ori, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 1)
        svx = np.flip(svx, 1)
        ori = np.flip(ori, 1)
        ori[:, :, 1] = - ori[:, :, 1]
    return sat, svx, ori


def random_vertical_flip(sat, svx, ori, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 0)
        svx = np.flip(svx, 0)
        ori = np.flip(ori, 0)
        ori[:, :, 0] = - ori[:, :, 0]
    return sat, svx, ori


def random_rotate_90(sat, svx, ori, u=0.5):
    if np.random.random() < u:
        sat = np.rot90(sat)
        svx = np.rot90(svx)
        ori = np.rot90(ori)
        ori[:, :, 1] = - ori[:, :, 1]
        ori = ori[:, :, [1, 0]]
    return sat, svx, ori


class TopoRoadDataset(data.Dataset):
    def __init__(self, root_dir):
        self.sample_list = list(map(lambda x: x[:-8], glob.glob(root_dir + '*_sat.png')))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sat = cv2.imread(os.path.join('{}_sat.png').format(self.sample_list[item]))
        svx = cv2.imread(os.path.join('{}_svx.png').format(self.sample_list[item]))
        ori = pickle.load(open(os.path.join('{}_ori.pkl').format(self.sample_list[item]), 'rb'))

        sat = random_hue_saturation_value(sat, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        sat, svx, ori = random_shift_scale_rotate(sat, svx, ori, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), aspect_limit=(-0, 0), rotate_limit=(-0, 0))
        sat, svx, ori = random_horizontal_flip(sat, svx, ori)
        sat, svx, ori = random_vertical_flip(sat, svx, ori)
        sat, svx, ori = random_rotate_90(sat, svx, ori)

        sat = torch.Tensor(np.array(sat, np.float32).transpose((2, 0, 1))) / 255.0 * 3.2 - 1.6
        seg = torch.Tensor(np.array(np.expand_dims(svx[:, :, 0], 2), np.float32).transpose((2, 0, 1))) / 255.0
        vex = torch.Tensor(np.array(svx[:, :, 1:], np.float32).transpose((2, 0, 1))) / 255.0
        ori = torch.Tensor(np.array(ori, np.float32).transpose((2, 0, 1)))

        return {'sat': sat, 'seg': seg, 'vex': vex, 'ori': ori}
