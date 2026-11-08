import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.registry import MODELS


def r_iou_loss(pred, target, beta=1.0, eps=1e-6, reduction='mean', avg_factor=None):
    if pred.size(0) == 0:
        return pred.sum() * 0

    pred = pred.clone()
    target = target.clone()
    pred[:, 4] = (pred[:, 4] + torch.pi / 4) % torch.pi - torch.pi / 4
    target[:, 4] = (target[:, 4] + torch.pi / 4) % torch.pi - torch.pi / 4

    center_dist = torch.sqrt(
        (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2 + eps
    )

    w_ratio = torch.clamp(1 - center_dist / (pred[:, 2] + eps), min=0, max=1)
    h_ratio = torch.clamp(1 - center_dist / (pred[:, 3] + eps), min=0, max=1)
    angle_diff = torch.abs(pred[:, 4] - target[:, 4])
    angle_ratio = torch.clamp(1 - angle_diff / (torch.pi / 2 + eps), min=0, max=1)

    approx_iou = w_ratio * h_ratio * angle_ratio

    iou_loss = F.smooth_l1_loss(
        approx_iou, torch.ones_like(approx_iou), beta=beta, reduction='none'
    )

    if reduction == 'mean':
        loss = weight_reduce_loss(iou_loss, None, reduction, avg_factor)
    elif reduction == 'sum':
        loss = iou_loss.sum()
    else:
        loss = iou_loss
    return loss


@MODELS.register_module()
class R_IoULoss(nn.Module):
    def __init__(self, beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        loss = r_iou_loss(
            pred, target, beta=self.beta, eps=self.eps,
            reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
