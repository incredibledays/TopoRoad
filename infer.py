import numpy as np
import cv2
import os
import math
import pickle
import json
import scipy
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters
import torch
from torch.autograd import Variable as V


def distance(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return np.sqrt(a * a + b * b)


def vNorm(v1):
    l = distance(v1, (0, 0)) + 0.0000001
    return v1[0] / l, v1[1] / l


def anglediff(v1, v2):
    v1 = vNorm(v1)
    v2 = vNorm(v2)
    return v1[0]*v2[0] + v1[1] * v2[1]


def cosine_similarity(k1, k2, k3):
    vec1 = distance_norm(k2, k1)
    vec2 = distance_norm(k3, k1)
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def distance_norm(k1, k2):
    l = distance(k1, k2)
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return a/l, b/l


def point2lineDistance(p, n1, n2):
    length = distance(n1, n2)

    v1 = [n1[0] - p[0], n1[1] - p[1]]
    v2 = [n2[0] - p[0], n2[1] - p[1]]

    area = abs(v1[0] * v2[1] - v1[1] * v2[0])

    if n1 == n2:
        return 0

    return area / length


def douglasPeucker(node_list, e=5.0):

    if len(node_list) <= 2:
        return node_list

    best_i = 1
    best_d = 0

    for i in range(1, len(node_list) - 1):
        d = point2lineDistance(node_list[i], node_list[0], node_list[-1])
        if d > best_d:
            best_d = d
            best_i = i

    if best_d <= e:
        return [node_list[0], node_list[-1]]

    new_list = douglasPeucker(node_list[0:best_i + 1], e=e)
    new_list = new_list[:-1] + douglasPeucker(node_list[best_i:len(node_list)], e=e)

    return new_list


def graphInsert(node_neighbor, n1key, n2key):
    if n1key != n2key:
        if n1key in node_neighbor:
            if n2key in node_neighbor[n1key]:
                pass
            else:
                node_neighbor[n1key].append(n2key)
        else:
            node_neighbor[n1key] = [n2key]

        if n2key in node_neighbor:
            if n1key in node_neighbor[n2key]:
                pass
            else:
                node_neighbor[n2key].append(n1key)
        else:
            node_neighbor[n2key] = [n1key]

    return node_neighbor


def simpilfyGraph(node_neighbor, e=2.5):

    visited = []

    new_node_neighbor = {}

    for node, node_nei in node_neighbor.items():
        if len(node_nei) == 1 or len(node_nei) > 2:
            if node in visited:
                continue

            # search node_nei

            for next_node in node_nei:
                if next_node in visited:
                    continue

                node_list = [node, next_node]

                while True:
                    if len(node_neighbor[node_list[-1]]) == 2:
                        if node_neighbor[node_list[-1]][0] == node_list[-2]:
                            node_list.append(node_neighbor[node_list[-1]][1])
                        else:
                            node_list.append(node_neighbor[node_list[-1]][0])
                    else:
                        break

                for i in range(len(node_list) - 1):
                    if node_list[i] not in visited:
                        visited.append(node_list[i])

                # simplify node_list
                new_node_list = douglasPeucker(node_list, e=e)

                for i in range(len(new_node_list) - 1):
                    new_node_neighbor = graphInsert(new_node_neighbor, new_node_list[i], new_node_list[i + 1])

    return new_node_neighbor


def graph_refine(graph, isolated_thr=150, spurs_thr=30):
    neighbors = graph

    gid = 0
    grouping = {}

    for k, v in neighbors.items():
        if k not in grouping:
            # start a search
            queue = [k]

            while len(queue) > 0:
                n = queue.pop(0)

                if n not in grouping:
                    grouping[n] = gid
                    for nei in neighbors[n]:
                        queue.append(nei)

            gid += 1

    group_count = {}

    for k, v in grouping.items():
        if v not in group_count:
            group_count[v] = (1, 0)
        else:
            group_count[v] = (group_count[v][0] + 1, group_count[v][1])

        for nei in neighbors[k]:
            a = k[0] - nei[0]
            b = k[1] - nei[1]

            d = np.sqrt(a * a + b * b)

            group_count[v] = (group_count[v][0], group_count[v][1] + d / 2)

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            if len(neighbors[v[0]]) >= 3:
                a = k[0] - v[0][0]
                b = k[1] - v[0][1]

                d = np.sqrt(a * a + b * b)

                if d < spurs_thr:
                    remove_list.append(k)

    remove_list2 = []
    remove_counter = 0
    new_neighbors = {}

    def isRemoved(k):
        gid = grouping[k]
        if group_count[gid][0] <= 1:
            return True
        elif group_count[gid][1] <= isolated_thr:
            return True
        elif k in remove_list:
            return True
        elif k in remove_list2:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    return new_neighbors


def graph_shave(graph, spurs_thr=50):
    neighbors = graph

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            d = distance(k, v[0])
            cur = v[0]
            l = [k]
            while True:
                if len(neighbors[cur]) >= 3:
                    break
                elif len(neighbors[cur]) == 1:
                    l.append(cur)
                    break

                else:

                    if neighbors[cur][0] == l[-1]:
                        next_node = neighbors[cur][1]
                    else:
                        next_node = neighbors[cur][0]

                    d += distance(cur, next_node)
                    l.append(cur)

                    cur = next_node

            if d < spurs_thr:
                for n in l:
                    if n not in remove_list:
                        remove_list.append(n)

    def isRemoved(k):
        if k in remove_list:
            return True
        else:
            return False

    new_neighbors = {}
    remove_counter = 0

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    # print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors


def graph_refine_deloop(neighbors, max_step=10, max_length=200, max_diff=5):
    removed = []
    impact = []

    remove_edge = []
    new_edge = []

    for k, v in neighbors.items():
        if k in removed:
            continue

        if k in impact:
            continue

        if len(v) < 2:
            continue

        for nei1 in v:
            if nei1 in impact:
                continue

            if k in impact:
                continue

            for nei2 in v:
                if nei2 in impact:
                    continue
                if nei1 == nei2:
                    continue

                if cosine_similarity(k, nei1, nei2) > 0.984:
                    l1 = distance(k, nei1)
                    l2 = distance(k, nei2)

                    # print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

                    if l2 < l1:
                        nei1, nei2 = nei2, nei1

                    remove_edge.append((k, nei2))
                    remove_edge.append((nei2, k))

                    new_edge.append((nei1, nei2))

                    impact.append(k)
                    impact.append(nei1)
                    impact.append(nei2)

                    break

    new_neighbors = {}

    def isRemoved(k):
        if k in removed:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                elif (nei, k) in remove_edge:
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    for new_e in new_edge:
        nk1 = new_e[0]
        nk2 = new_e[1]

        if nk2 not in new_neighbors[nk1]:
            new_neighbors[nk1].append(nk2)
        if nk1 not in new_neighbors[nk2]:
            new_neighbors[nk2].append(nk1)

    # print("remove %d edges" % len(remove_edge))

    return new_neighbors, len(remove_edge)


def detect_local_minima(arr, mask, threshold=0.5):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    return np.where((detected_minima & (mask > threshold)))


def detect_keypoints(ach, v_thr):
    kp = np.copy(ach)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp), 0.001)
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, v_thr)
    return keypoints


def line_pooling(src, x0, y0, x, y, ori=False):
    step = round(math.sqrt((x - x0) ** 2 + (y - y0) ** 2))
    sample = np.linspace(np.array([x0, y0]), np.array([x, y]), step, dtype=int)
    if ori:
        mean = (np.mean(src[sample[round(step / 8):round(3 * step / 8), 0], sample[round(step / 8):round(3 * step / 8), 1]]) + np.mean(src[sample[round(5 * step / 8):round(7 * step / 8), 0], sample[round(5 * step / 8):round(7 * step / 8), 1]])) / 2
        std = (np.std(src[sample[round(step / 8):round(3 * step / 8), 0], sample[round(step / 8):round(3 * step / 8), 1]]) + np.std(src[sample[round(5 * step / 8):round(7 * step / 8), 0], sample[round(5 * step / 8):round(7 * step / 8), 1]])) / 2
        return mean, std
    else:
        mean = np.mean(src[sample[round(step / 4):round(3 * step / 4), 0], sample[round(step / 4):round(3 * step / 4), 1]])
        std = np.std(src[sample[round(step / 4):round(3 * step / 4), 0], sample[round(step / 4):round(3 * step / 4), 1]])
        return mean, std


def DecodeRoadGraphSVPO(seg, vex, ori, rad):
    # seg_ori = cv2.erode(np.sqrt(ori[:, :, 0] ** 2 + ori[:, :, 1] ** 2), np.ones((5, 5), np.uint8))
    # seg = cv2.erode(seg, np.ones((3, 3), np.uint8))
    vertices = detect_keypoints(vex[:, :, 0], 0.05)
    candidates = []
    for j in range(0, len(vertices[0])):
        x0 = vertices[0][j]
        y0 = vertices[1][j]
        if x0 == -1 and y0 == -1:
            continue
        z0 = round(vex[x0, y0, 1] * 8)
        for k in range(j + 1, len(vertices[0])):
            x = vertices[0][k]
            y = vertices[1][k]
            if x == -1 and y == -1:
                continue
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                vertices[0][k] = -1
                vertices[1][k] = -1
                x0 = (x0 + x) // 2
                y0 = (y0 + y) // 2
                z0 = max(z0, round(vex[x, y, 1] * 8))
        candidates.append([x0, y0, z0])

    neighbors = {}
    for j in range(len(candidates)):
        x0 = candidates[j][0]
        y0 = candidates[j][1]
        z0 = candidates[j][2]
        proposals = []
        for k in range(len(candidates)):
            if j == k:
                continue
            x = candidates[k][0]
            y = candidates[k][1]
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue

            line_mean, line_std = line_pooling(seg, x0, y0, x, y)

            if line_mean < 0.5 or line_std > 0.18:
                continue

            mean_x, _ = line_pooling(ori[:, :, 0], x0, y0, (x0 + x) // 2, (y0 + y) // 2)
            mean_y, _ = line_pooling(ori[:, :, 1], x0, y0, (x0 + x) // 2, (y0 + y) // 2)
            angle = math.atan2(y - y0, x - x0)
            angle_pre = math.atan2(mean_y, mean_x)
            delta_pre = abs(angle - angle_pre)
            if delta_pre > math.pi:
                delta_pre = 2 * math.pi - delta_pre
            if delta_pre > (math.pi / 2):
                continue

            # angle = math.atan2(y - y0, x - x0)
            dist = distance((x0, y0), (x, y))
            proposals.append([x, y, dist, angle, line_mean])
            for point in proposals[:-1]:
                delta = abs(angle - point[3])
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                if delta < (math.pi / 16):
                    if dist < point[2]:
                        proposals.remove(point)
                    else:
                        proposals.remove([x, y, dist, angle, line_mean])
                    break

        proposals.sort(key=lambda t: t[4], reverse=True)
        for point in proposals[:z0]:
            if (x0, y0) in neighbors:
                if (point[0], point[1]) in neighbors[(x0, y0)]:
                    pass
                else:
                    neighbors[(x0, y0)].append((point[0], point[1]))
            else:
                neighbors[(x0, y0)] = [(point[0], point[1])]

            if (point[0], point[1]) in neighbors:
                if (x0, y0) in neighbors[(point[0], point[1])]:
                    pass
                else:
                    neighbors[(point[0], point[1])].append((x0, y0))
            else:
                neighbors[(point[0], point[1])] = [(x0, y0)]

    spurs_thr = 50
    isolated_thr = 200
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def infer_cityscale():
    from extractor import Extractor as Extractor
    from network import SegVexPlusOriDLA as Net

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/cityscale/'
    weight_dir = './checkpoints/cityscale/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(Net, eval_mode=True)
    model.load(weight_dir)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        vex_pre = pre['vex'].squeeze().cpu().data.numpy().transpose((1, 2, 0))
        ori_pre = pre['ori'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        pre = None

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        # cv2.imwrite('./seg.jpg', seg_pre * 255)
        # cv2.imwrite('./vex.jpg', vex_pre[:, :, 0] * 255)
        # seg_ori = np.sqrt(ori_pre[:, :, 0] ** 2 + ori_pre[:, :, 1] ** 2)
        # seg_ori = cv2.erode(seg_ori, np.ones((5, 5), np.uint8))
        # seg = np.clip(seg_ori, a_min=0, a_max=1)
        # cv2.imwrite('./seg.jpg', seg * 255)
        # ori_pre = (ori_pre / 2 + 0.5) * 255
        # seg_ori = np.concatenate([np.expand_dims(seg_ori, 2), np.expand_dims(seg_ori * ori_pre[:, :, 0], 2), np.expand_dims(seg_ori * ori_pre[:, :, 1], 2)], 2)
        # cv2.imwrite('./seg_ori.jpg', seg_ori)
        # return

        graph = DecodeRoadGraphSVPO(seg_pre, vex_pre, ori_pre, 75)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        # graph = simpilfyGraph(graph)
        seg = np.zeros_like(seg_pre)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (0, 128, 255), 3)
                cv2.line(seg, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), 255, 3)
        for u, v in graph.items():
            n1 = u
            if len(v) < 3:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 128), -1)
        cv2.imwrite(output_dir + "region_%d_vis.png" % i, sat)
        cv2.imwrite(output_dir + "region_%d_seg.png" % i, seg)


def DecodeRoadGraphSVPO2(seg, vex, ori, rad):
    # seg_ori = cv2.erode(np.sqrt(ori[:, :, 0] ** 2 + ori[:, :, 1] ** 2), np.ones((5, 5), np.uint8))
    # seg = cv2.erode(seg, np.ones((3, 3), np.uint8))
    vertices = detect_keypoints(vex[:, :, 0], 0.05)
    candidates = []
    for j in range(0, len(vertices[0])):
        x0 = vertices[0][j]
        y0 = vertices[1][j]
        if x0 == -1 and y0 == -1:
            continue
        z0 = round(vex[x0, y0, 1] * 8)
        for k in range(j + 1, len(vertices[0])):
            x = vertices[0][k]
            y = vertices[1][k]
            if x == -1 and y == -1:
                continue
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                vertices[0][k] = -1
                vertices[1][k] = -1
                x0 = (x0 + x) // 2
                y0 = (y0 + y) // 2
                z0 = max(z0, round(vex[x, y, 1] * 8))
        candidates.append([x0, y0, z0])

    neighbors = {}
    for j in range(len(candidates)):
        x0 = candidates[j][0]
        y0 = candidates[j][1]
        z0 = candidates[j][2]
        proposals = []
        for k in range(len(candidates)):
            if j == k:
                continue
            x = candidates[k][0]
            y = candidates[k][1]
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue

            line_mean, line_std = line_pooling(seg, x0, y0, x, y)
            # line_mean_ori, line_std_ori = line_pooling(seg_ori, x0, y0, x, y, True)
            if line_mean < 0.2 or line_std > 0.3:
                continue

            mean_x, _ = line_pooling(ori[:, :, 0], x0, y0, (x0 + x) // 2, (y0 + y) // 2)
            mean_y, _ = line_pooling(ori[:, :, 1], x0, y0, (x0 + x) // 2, (y0 + y) // 2)
            angle = math.atan2(y - y0, x - x0)
            angle_pre = math.atan2(mean_y, mean_x)
            delta_pre = abs(angle - angle_pre)
            if delta_pre > math.pi:
                delta_pre = 2 * math.pi - delta_pre
            if delta_pre > (math.pi / 4):
                continue

            # angle = math.atan2(y - y0, x - x0)
            dist = distance((x0, y0), (x, y))
            proposals.append([x, y, dist, angle, line_mean])
            for point in proposals[:-1]:
                delta = abs(angle - point[3])
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                if delta < (math.pi / 32):
                    if dist < point[2]:
                        proposals.remove(point)
                    else:
                        proposals.remove([x, y, dist, angle, line_mean])
                    break

        proposals.sort(key=lambda t: t[4], reverse=True)
        for point in proposals[:z0]:
            if (x0, y0) in neighbors:
                if (point[0], point[1]) in neighbors[(x0, y0)]:
                    pass
                else:
                    neighbors[(x0, y0)].append((point[0], point[1]))
            else:
                neighbors[(x0, y0)] = [(point[0], point[1])]

            if (point[0], point[1]) in neighbors:
                if (x0, y0) in neighbors[(point[0], point[1])]:
                    pass
                else:
                    neighbors[(point[0], point[1])].append((x0, y0))
            else:
                neighbors[(point[0], point[1])] = [(x0, y0)]

    spurs_thr = 25
    isolated_thr = 100
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=0, spurs_thr=0))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def infer_spacenet():
    from extractor import Extractor as Extractor
    from network import SegVexPlusOriDLA as Net

    input_dir = './datasets/spacenet/test/'
    output_dir = './results/spacenet/'
    weight_dir = './checkpoints/spacenet/best3.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(Net, eval_mode=True)
    model.load(weight_dir)

    dataset = json.load(open('./datasets/spacenet/dataset.json', 'r'))
    for item in dataset['test']:
        sat = cv2.imread(input_dir + item + '_sat.png')
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        vex_pre = pre['vex'].squeeze().cpu().data.numpy().transpose((1, 2, 0))
        ori_pre = pre['ori'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        pre = None

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()

        pre = None

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        # cv2.imwrite('./seg.jpg', seg_pre * 255)
        # cv2.imwrite('./vex.jpg', vex_pre[:, :, 0] * 255)
        # seg_ori = np.sqrt(ori_pre[:, :, 0] ** 2 + ori_pre[:, :, 1] ** 2)
        # seg_ori = cv2.erode(seg_ori, np.ones((5, 5), np.uint8))
        # seg = np.clip(seg_ori, a_min=0, a_max=1)
        # cv2.imwrite('./seg.jpg', seg * 255)
        # ori_pre = (ori_pre / 2 + 0.5) * 255
        # seg_ori = np.concatenate([np.expand_dims(seg_ori, 2), np.expand_dims(seg_ori * ori_pre[:, :, 0], 2), np.expand_dims(seg_ori * ori_pre[:, :, 1], 2)], 2)
        # cv2.imwrite('./seg_ori.jpg', seg_ori)
        # return

        graph = DecodeRoadGraphSVPO2(seg_pre, vex_pre, ori_pre, 75)
        pickle.dump(graph, open(output_dir + "{}_graph.p".format(item), "wb"))
        # graph = simpilfyGraph(graph)
        seg = np.zeros_like(seg_pre)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (0, 128, 255), 3)
                cv2.line(seg, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), 255, 3)
        for u, v in graph.items():
            n1 = u
            if len(v) < 3:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 128), -1)
        cv2.imwrite(output_dir + "{}_vis.png".format(item), sat)
        cv2.imwrite(output_dir + "{}_seg.png".format(item), seg)
