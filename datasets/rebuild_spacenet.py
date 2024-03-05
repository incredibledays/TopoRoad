import cv2
import pickle
import json
import numpy as np
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


dataset = json.load(open('./spacenet/dataset.json', 'r'))
dataset_image_size = 400
rebuild_image_size = 512

for item in dataset['train']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))

    vex = np.zeros((rebuild_image_size, rebuild_image_size, 2))
    ori_x = np.zeros((rebuild_image_size, rebuild_image_size))
    ori_y = np.zeros((rebuild_image_size, rebuild_image_size))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 0] = np.ones((3, 3)) * 255
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255 * len(n_locs) / 8

        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            angle_norm = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            angle_x = (x - x0) / angle_norm
            angle_y = (y - y0) / angle_norm
            cv2.line(ori_x, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_x, 2)
            cv2.line(ori_y, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_y, 2)

    seg = np.expand_dims((ori_x ** 2 + ori_y ** 2) > 0, 2) * 255
    ori_x = np.expand_dims(ori_x, 2)
    ori_y = np.expand_dims(ori_y, 2)
    svx = np.concatenate([seg, vex], 2)
    ori = np.concatenate([ori_x, ori_y], 2)

    cv2.imwrite('./spacenet/train/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/train/' + '{}_svx.png'.format(item), svx)
    pickle.dump(ori, open('./spacenet/train/' + '{}_ori.pkl'.format(item), 'wb'))


for item in dataset['validation']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))

    vex = np.zeros((rebuild_image_size, rebuild_image_size, 2))
    ori_x = np.zeros((rebuild_image_size, rebuild_image_size))
    ori_y = np.zeros((rebuild_image_size, rebuild_image_size))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 0] = np.ones((3, 3)) * 255
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255 * len(n_locs) / 8

        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            angle_norm = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            angle_x = (x - x0) / angle_norm
            angle_y = (y - y0) / angle_norm
            cv2.line(ori_x, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_x, 2)
            cv2.line(ori_y, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_y, 2)

    seg = np.expand_dims((ori_x ** 2 + ori_y ** 2) > 0, 2) * 255
    ori_x = np.expand_dims(ori_x, 2)
    ori_y = np.expand_dims(ori_y, 2)
    svx = np.concatenate([seg, vex], 2)
    ori = np.concatenate([ori_x, ori_y], 2)

    cv2.imwrite('./spacenet/valid/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/valid/' + '{}_svx.png'.format(item), svx)
    pickle.dump(ori, open('./spacenet/valid/' + '{}_ori.pkl'.format(item), 'wb'))


for item in dataset['test']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))
    rebuild_neighbors = {}

    vex = np.zeros((rebuild_image_size, rebuild_image_size, 2))
    ori_x = np.zeros((rebuild_image_size, rebuild_image_size))
    ori_y = np.zeros((rebuild_image_size, rebuild_image_size))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 0] = np.ones((3, 3)) * 255
        vex[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255 * len(n_locs) / 8

        rebuild_neighbors[(x0, y0)] = []

        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            angle_norm = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            angle_x = (x - x0) / angle_norm
            angle_y = (y - y0) / angle_norm
            cv2.line(ori_x, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_x, 2)
            cv2.line(ori_y, (y0, x0), (round((y + y0) / 2), round((x + x0) / 2)), angle_y, 2)

            rebuild_neighbors[(x0, y0)].append((x, y))

    seg = np.expand_dims((ori_x ** 2 + ori_y ** 2) > 0, 2) * 255
    svx = np.concatenate([seg, vex], 2)

    cv2.imwrite('./spacenet/test/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/test/' + '{}_svx.png'.format(item), svx)
    pickle.dump(rebuild_neighbors, open('./spacenet/test/' + '{}__gt_graph_rebuild.p'.format(item), 'wb'))
