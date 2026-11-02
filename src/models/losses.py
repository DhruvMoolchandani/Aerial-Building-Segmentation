import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DiceBCELoss(nn.Module):
    """Kept as-is, used internally for the mask channel."""
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.dice(preds, targets) + self.bce(preds, targets)


class MaskBoundaryLoss(nn.Module):
    """
    preds: (B, 2, H, W) raw logits — channel 0 = mask, channel 1 = boundary
    targets: (B, 2, H, W) — channel 0 = mask GT, channel 1 = boundary GT
    """
    def __init__(self, boundary_loss_weight=2.0):
        super().__init__()
        self.mask_loss_fn = DiceBCELoss()
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()
        self.boundary_loss_weight = boundary_loss_weight

    def forward(self, preds, targets):
        mask_pred, boundary_pred = preds[:, 0:1], preds[:, 1:2]
        mask_gt, boundary_gt = targets[:, 0:1], targets[:, 1:2]

        mask_loss = self.mask_loss_fn(mask_pred, mask_gt)
        boundary_loss = self.boundary_loss_fn(boundary_pred, boundary_gt)

        return mask_loss + self.boundary_loss_weight * boundary_loss