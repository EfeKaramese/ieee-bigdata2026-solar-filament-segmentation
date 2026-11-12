"""Panoptic Quality exactly as the organisers compute it: every annotator record is scored separately against the
image's predictions and the counts are pooled. A (GT, prediction) pair is a TP when IoU > 0.5; predictions do not
overlap, so this equals standard PQ matching."""
import numpy as np

IOU_THR = 0.5


def iou_matrix(gts, lab, n):
    """(n_gt, n_pred) IoU between GT masks (flat index arrays) and the components of a label image."""
    if not gts or n == 0:
        return np.zeros((len(gts), n))
    flat = lab.ravel()
    area_p = np.bincount(flat, minlength=n + 1)[1:]
    out = np.zeros((len(gts), n))
    for i, g in enumerate(gts):
        inter = np.bincount(flat[g], minlength=n + 1)[1:]
        out[i] = inter / np.maximum(len(g) + area_p - inter, 1)
    return out


def record_score(iou, keep):
    """(sum of TP IoU, TP, FP, FN) for one record and the kept predictions."""
    iou = iou[:, keep]
    n_gt, n_p = iou.shape
    if n_gt == 0:
        return (0.0, 0, n_p, 0)
    if n_p == 0:
        return (0.0, 0, 0, n_gt)
    hit = iou > IOU_THR
    return (float(iou[hit].sum()), int(hit.sum()), int((~hit.any(0)).sum()), int((~hit.any(1)).sum()))


def image_parts(mats, keep):
    return np.sum([record_score(m, keep) for m in mats], 0) if mats else np.zeros(4)


def summary(parts):
    s, tp, fp, fn = np.asarray(parts, float).reshape(-1, 4).sum(0)
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=s / den if den else 0.0, SQ=s / tp if tp else 0.0, RQ=tp / den if den else 0.0,
                TP=int(tp), FP=int(fp), FN=int(fn))


def pq(parts):
    return summary(parts)["PQ"]


def paired_bootstrap(A, B, n=1000, seed=42):
    """PQ(A) - PQ(B) with a 95% CI, resampling images (A, B: images x 4 parts in the same image order)."""
    idx = np.random.default_rng(seed).integers(0, len(A), (n, len(A)))
    dist = np.array([pq(A[i]) - pq(B[i]) for i in idx])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return dict(diff=pq(A) - pq(B), lo=lo, hi=hi, p_better=float((dist > 0).mean()))
