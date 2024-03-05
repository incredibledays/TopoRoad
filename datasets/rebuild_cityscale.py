import pickle
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import math


def neighbor_to_integer(n_in):
    n_out = {}
    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))
        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []
        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))
            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)
        n_out[nk] = nv
    return n_out


input_dir = './cityscale/'
output_root = './cityscale/'
output_dir = ''
dataset_image_size = 2048
size = 512
stride = 256


# for i in tqdm(range(180)):
#     sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
#     neighbors = neighbor_to_integer(pickle.load(open(input_dir + 'region_%d_refine_gt_graph.p' % i, 'rb')))
#
#     ach = np.zeros((dataset_image_size, dataset_image_size, 2))
#     ori_x = np.zeros((dataset_image_size, dataset_image_size))
#     ori_y = np.zeros((dataset_image_size, dataset_image_size))
#     for loc, n_locs in neighbors.items():
#         if len(n_locs) < 3:
#             ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 0] = np.ones((3, 3)) * 255
#         else:
#             ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 1] = np.ones((3, 3)) * 255
#
#         for n_loc in n_locs:
#             angle_norm = math.sqrt((n_loc[0] - loc[0]) ** 2 + (n_loc[1] - loc[1]) ** 2)
#             angle_x = (n_loc[0] - loc[0]) / angle_norm
#             angle_y = (n_loc[1] - loc[1]) / angle_norm
#             cv2.line(ori_x, (loc[1], loc[0]), (round((n_loc[1] + loc[1]) / 2), round((n_loc[0] + loc[0]) / 2)), angle_x, 2)
#             cv2.line(ori_y, (loc[1], loc[0]), (round((n_loc[1] + loc[1]) / 2), round((n_loc[0] + loc[0]) / 2)), angle_y, 2)
#
#     seg = np.expand_dims((ori_x ** 2 + ori_y ** 2) > 0, 2) * 255
#     ori_x = np.expand_dims(ori_x, 2)
#     ori_y = np.expand_dims(ori_y, 2)
#     sap = np.concatenate([seg, ach], 2)
#     ori = np.concatenate([ori_x, ori_y], 2)
#
#     if i % 10 < 8:
#         output_dir = output_root + 'train/'
#     if i % 20 == 18:
#         output_dir = output_root + 'valid/'
#     if i % 20 == 8 or i % 10 == 9:
#         output_dir = output_root + 'test/'
#         shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
#         shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
#         cv2.imwrite(output_dir + 'region_%d_sap.png' % i, sap)
#         continue
#
#     maxx = int((dataset_image_size - size) / stride)
#     maxy = int((dataset_image_size - size) / stride)
#     for x in range(maxx + 1):
#         for y in range(maxy + 1):
#             sat_block = sat[x * stride:x * stride + size, y * stride:y * stride + size, :]
#             sap_block = sap[x * stride:x * stride + size, y * stride:y * stride + size, :]
#             ori_block = ori[x * stride:x * stride + size, y * stride:y * stride + size, :]
#             cv2.imwrite(output_dir + '{}_{}_{}_sat.png'.format(i, x, y), sat_block)
#             cv2.imwrite(output_dir + '{}_{}_{}_sap.png'.format(i, x, y), sap_block)
#             pickle.dump(ori_block, open(output_dir + '{}_{}_{}_ori.pkl'.format(i, x, y), 'wb'))


for i in tqdm(range(180)):
    sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
    neighbors = neighbor_to_integer(pickle.load(open(input_dir + 'region_%d_refine_gt_graph.p' % i, 'rb')))

    vex = np.zeros((dataset_image_size, dataset_image_size, 2))
    ori_x = np.zeros((dataset_image_size, dataset_image_size))
    ori_y = np.zeros((dataset_image_size, dataset_image_size))
    for loc, n_locs in neighbors.items():
        vex[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 0] = np.ones((3, 3)) * 255
        vex[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 1] = np.ones((3, 3)) * 255 * len(n_locs) / 8

        for n_loc in n_locs:
            angle_norm = math.sqrt((n_loc[0] - loc[0]) ** 2 + (n_loc[1] - loc[1]) ** 2)
            angle_x = (n_loc[0] - loc[0]) / angle_norm
            angle_y = (n_loc[1] - loc[1]) / angle_norm
            cv2.line(ori_x, (loc[1], loc[0]), (round((n_loc[1] + loc[1]) / 2), round((n_loc[0] + loc[0]) / 2)), angle_x, 2)
            cv2.line(ori_y, (loc[1], loc[0]), (round((n_loc[1] + loc[1]) / 2), round((n_loc[0] + loc[0]) / 2)), angle_y, 2)

    seg = np.expand_dims((ori_x ** 2 + ori_y ** 2) > 0, 2) * 255
    ori_x = np.expand_dims(ori_x, 2)
    ori_y = np.expand_dims(ori_y, 2)
    svx = np.concatenate([seg, vex], 2)
    ori = np.concatenate([ori_x, ori_y], 2)

    if i % 10 < 8:
        output_dir = output_root + 'train/'
    if i % 20 == 18:
        output_dir = output_root + 'valid/'
    if i % 20 == 8 or i % 10 == 9:
        output_dir = output_root + 'test/'
        shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
        shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
        cv2.imwrite(output_dir + 'region_%d_svx.png' % i, svx)
        continue

    maxx = int((dataset_image_size - size) / stride)
    maxy = int((dataset_image_size - size) / stride)
    for x in range(maxx + 1):
        for y in range(maxy + 1):
            sat_block = sat[x * stride:x * stride + size, y * stride:y * stride + size, :]
            svx_block = svx[x * stride:x * stride + size, y * stride:y * stride + size, :]
            ori_block = ori[x * stride:x * stride + size, y * stride:y * stride + size, :]
            cv2.imwrite(output_dir + '{}_{}_{}_sat.png'.format(i, x, y), sat_block)
            cv2.imwrite(output_dir + '{}_{}_{}_svx.png'.format(i, x, y), svx_block)
            pickle.dump(ori_block, open(output_dir + '{}_{}_{}_ori.pkl'.format(i, x, y), 'wb'))
