import cv2
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as data


def train_cityscale():
    """DLASeg as backbone, including segmentation branch, vertex plus branch, and orientation branch."""
    from dataset import TopoRoadDataset as Dataset
    from extractor import Extractor as Extractor
    from network import SegVexPlusOriDLA as Net
    from loss import SegVexPlusOriLoss as Loss

    data_dir = './datasets/cityscale/train/'
    checkpoint_dir = './checkpoints/cityscale/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 8
    num_workers = 4
    total_epoch = 100

    dataset = Dataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(Net, Loss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        vex_vis = torch.cat((pre['seg'][0], pre['vex'][0][0].unsqueeze(0), pre['vex'][0][1].unsqueeze(0)), 0) * 255
        ori_vis = torch.cat((pre['seg'][0], pre['seg'][0] * (pre['ori'][0][0].unsqueeze(0) / 2 + 0.5), pre['seg'][0] * (pre['ori'][0][1].unsqueeze(0) / 2 + 0.5)), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_vex.jpg'.format(epoch), np.uint8(vex_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_ori.jpg'.format(epoch), np.uint8(ori_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        if epoch % 10 == 0:
            model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()


def train_spacenet():
    """DLASeg as backbone, including segmentation branch, vertex plus branch, and orientation branch for spacenet dataset."""
    from dataset import TopoRoadDataset as Dataset
    from extractor import Extractor as Extractor
    from network import SegVexPlusOriDLA as Net
    from loss import SegVexPlusOriLoss as Loss

    data_dir = './datasets/spacenet/train/'
    checkpoint_dir = './checkpoints/spacenet/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 8
    num_workers = 4
    total_epoch = 100

    dataset = Dataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(Net, Loss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        vex_vis = torch.cat((pre['seg'][0], pre['vex'][0][0].unsqueeze(0), pre['vex'][0][1].unsqueeze(0)), 0) * 255
        ori_vis = torch.cat((pre['seg'][0], pre['seg'][0] * (pre['ori'][0][0].unsqueeze(0) / 2 + 0.5), pre['seg'][0] * (pre['ori'][0][1].unsqueeze(0) / 2 + 0.5)), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_vex.jpg'.format(epoch), np.uint8(vex_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_ori.jpg'.format(epoch), np.uint8(ori_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        if epoch % 10 == 0:
            model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()
