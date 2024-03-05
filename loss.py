import torch
import torch.nn as nn


class SegVexPlusOriLoss(nn.Module):
    def __init__(self):
        super(SegVexPlusOriLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def __call__(self, pre, gt):
        seg_loss = self.bce_loss(pre['seg'], gt['seg'])
        vex_loss = self.bce_loss(pre['vex'][:, 0, :, :], gt['vex'][:, 0, :, :]) * 10
        vex_mask = torch.clip(gt['vex'][:, 0, :, :] + 0.01, 0, 1)
        num_loss = torch.sum(self.smooth_l1(pre['vex'][:, 1, :, :], gt['vex'][:, 1, :, :]) * vex_mask) / torch.sum(vex_mask) * 100
        seg_mask = torch.clip(gt['seg'] + 0.01, 0, 1)
        ori_loss = torch.sum(self.smooth_l1(pre['ori'], gt['ori']) * seg_mask) / torch.sum(seg_mask)
        print(seg_loss.item(), vex_loss.item(), num_loss.item(), ori_loss.item())
        return seg_loss + vex_loss + num_loss + ori_loss
