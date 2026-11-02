import segmentation_models_pytorch as smp
import torch


def get_batch_stats(preds, targets, threshold=0.5):
    """Returns raw tp, fp, fn, tn tensors for one batch (not yet divided into a ratio)."""
    preds = (torch.sigmoid(preds) > threshold).long()
    targets = targets.long()
    tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, mode="binary")
    return tp, fp, fn, tn


def compute_metrics_from_stats(tp, fp, fn, tn):
    """Computes final IoU/Dice/Precision/Recall from accumulated stats (call once, at the end)."""
    return {
        "iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
        "dice": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
        "precision": smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
        "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
    }


# Kept for any code that still wants quick per-batch numbers (e.g. ad-hoc debugging) —
# not used for the official validation loop anymore.
def compute_metrics(preds, targets, threshold=0.5):
    tp, fp, fn, tn = get_batch_stats(preds, targets, threshold)
    return compute_metrics_from_stats(tp, fp, fn, tn)