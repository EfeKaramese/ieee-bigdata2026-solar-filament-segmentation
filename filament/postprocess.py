"""From a probability map to scored filament instances.

1. Components: hysteresis (grow at t_lo from seeds above t_hi), morphological closing, hole filling, minimum area.
2. 31 features per component (geometry, probability statistics, contrast, context, skeleton).
3. Scorer: gradient-boosted regressor predicting u = the component's expected matched IoU over the image's records
   (0 when it would be a false positive). Keeping a component with expected IoU u adds u to the PQ numerator and 1/2 to
   the denominator, so it pays off when u > PQ/2; the threshold tau is tuned on out-of-fold predictions.
"""
import math

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.ensemble import HistGradientBoostingRegressor

from .evaluate import IOU_THR

EIGHT = np.ones((3, 3), bool)
STATION = {c: i for i, c in enumerate("BCLMTU")}
BASE_NAMES = ["log_area", "p_mean", "p_max", "p_min", "p_std", "x_mean", "x_ring", "x_contrast", "log_elongation",
              "bbox_fill", "log_bbox_h", "log_bbox_w", "radial_pos", "area_rank", "n_components", "disk_p_mean",
              "station", "n_pixels_p05", "area_frac"]
EXTRA_NAMES = ["frac_p05", "frac_p07", "frac_p09", "skeleton_len", "skeleton_p_mean", "skeleton_p_min", "mean_width",
               "ring_p_mean", "ring_p_max", "nearest_dist", "neighbours_150", "local_texture", "skeleton_ends"]


def components(prob, t_lo, t_hi, close, min_area):
    """Label image (int32) and number of components."""
    lab, n = ndi.label(prob >= t_lo, EIGHT)
    if n == 0:
        return lab.astype(np.int32), 0
    seeds = np.unique(lab[prob >= t_hi])
    m = np.isin(lab, seeds[seeds > 0])
    if close > 0 and m.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lab, n = ndi.label(ndi.binary_fill_holes(m), EIGHT)
    if n == 0:
        return lab.astype(np.int32), 0
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = np.flatnonzero(area >= min_area) + 1
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[lab], len(keep)


def components_cfg(prob, cfg):
    p = cfg["postprocess"]
    lab, n = components(prob, p["t_lo"], p["t_hi"], p["close"], p["min_area"])
    if n == 0 and prob.max() > 0:      # never return an empty image: fall back to relative thresholds
        lab, n = components(prob, 0.5 * prob.max(), 0.8 * prob.max(), p["close"], 1)
    return lab, n


def base_features(lab, n, prob, x, stem, full=2048):
    if n == 0:
        return np.zeros((0, 19), np.float32)
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(prob), lab, idx)
    p_mean, p_max, p_min = ndi.mean(prob, lab, idx), ndi.maximum(prob, lab, idx), ndi.minimum(prob, lab, idx)
    p_std = np.sqrt(np.maximum(ndi.mean(prob ** 2, lab, idx) - p_mean ** 2, 0))
    x_mean = ndi.mean(x, lab, idx)
    ring = ndi.grey_dilation(lab, size=(9, 9))
    ring[lab > 0] = 0
    x_ring = ndi.mean(x, ring, idx)
    x_ring = np.where(np.isfinite(x_ring), x_ring, x_mean)
    yy, xx = np.indices(lab.shape)
    my, mx = ndi.mean(yy, lab, idx), ndi.mean(xx, lab, idx)
    vyy = ndi.mean(yy ** 2, lab, idx) - my ** 2
    vxx = ndi.mean(xx ** 2, lab, idx) - mx ** 2
    vxy = ndi.mean(yy * xx, lab, idx) - my * mx
    tr = vyy + vxx
    disc = np.sqrt(np.maximum((vyy - vxx) ** 2 + 4 * vxy ** 2, 0))
    elong = np.sqrt((tr + disc) / np.maximum(tr - disc, 1e-6))
    sl = ndi.find_objects(lab)
    bh = np.array([q[0].stop - q[0].start for q in sl])
    bw = np.array([q[1].stop - q[1].start for q in sl])
    disk = x > 0
    dy, dx = np.nonzero(disk)
    cy, cx = (dy.mean(), dx.mean()) if len(dy) else (lab.shape[0] / 2, lab.shape[1] / 2)
    R = math.sqrt(disk.sum() / math.pi)
    return np.column_stack([
        np.log1p(area), p_mean, p_max, p_min, p_std, x_mean, x_ring, x_ring - x_mean, np.log1p(elong),
        area / np.maximum(bh * bw, 1), np.log1p(bh), np.log1p(bw), np.hypot(my - cy, mx - cx) / max(R, 1),
        np.argsort(np.argsort(-area)) / max(n - 1, 1), np.full(n, n),
        np.full(n, float(prob[disk].mean()) if disk.any() else 0.0), np.full(n, STATION.get(stem[-2], -1)),
        np.full(n, float((prob > 0.5).sum())), area / (full * full) * 1e4]).astype(np.float32)


def extra_features(lab, n, prob, x, scale=1.0):
    """scale = working size / 2048 (distances are expressed in 2048 px)."""
    if n == 0:
        return np.zeros((0, len(EXTRA_NAMES)), np.float32)
    idx = np.arange(1, n + 1)
    out = np.zeros((n, len(EXTRA_NAMES)), np.float32)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float32)
    for k, t in enumerate((0.5, 0.7, 0.9)):
        out[:, k] = np.bincount(lab.ravel(), weights=(prob >= t).ravel(), minlength=n + 1)[1:] / np.maximum(area, 1)
    ring = ndi.grey_dilation(lab, size=(int(9 * scale) | 1,) * 2)
    ring[lab > 0] = 0
    out[:, 7] = np.nan_to_num(ndi.mean(prob, ring, idx))
    out[:, 8] = np.nan_to_num(ndi.maximum(prob, ring, idx))
    yy, xx = np.indices(lab.shape)
    C = np.column_stack([ndi.mean(xx, lab, idx), ndi.mean(yy, lab, idx)])
    if n > 1:
        D = np.hypot(*(C[:, None, :] - C[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        out[:, 9] = D.min(1) / scale
        out[:, 10] = (D < 150 * scale).sum(1)
    else:
        out[:, 9] = 2048
        out[:, 10] = 0
    tex = cv2.GaussianBlur(x, (0, 0), 3 * scale)
    tex = np.sqrt(np.maximum(cv2.GaussianBlur((x - tex) ** 2, (0, 0), 6 * scale), 0))
    out[:, 11] = np.nan_to_num(ndi.mean(tex, ring, idx))
    for k, s in enumerate(ndi.find_objects(lab)):
        m = lab[s] == k + 1
        sk = skeletonize(m)
        L = int(sk.sum())
        out[k, 3] = L / scale
        if L:
            ps = prob[s][sk]
            out[k, 4], out[k, 5] = ps.mean(), ps.min()
            nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
            out[k, 12] = int(((nb == 2) & sk).sum())
        out[k, 6] = area[k] / max(L, 1) / scale
    return out


def feature_matrix(X19, X13, station=False):
    X = np.hstack([X19, X13])
    return X if station else np.delete(X, BASE_NAMES.index("station"), axis=1)


def feature_names(station=False):
    names = BASE_NAMES + EXTRA_NAMES
    return names if station else [n for n in names if n != "station"]


def target_u(mats, n):
    """Mean over records of the component's matched IoU (0 where it matches nothing)."""
    acc = np.zeros(n)
    for m in mats:
        if m.shape[0]:
            acc += (m * (m > IOU_THR)).max(0)
    return acc / max(len(mats), 1)


def make_regressor(cfg, seed):
    s = cfg["scorer"]
    return HistGradientBoostingRegressor(max_iter=s["max_iter"], learning_rate=s["learning_rate"],
                                         max_leaf_nodes=s["max_leaf_nodes"], l2_regularization=s["l2_regularization"],
                                         random_state=seed)


def crossfit(X, Y, groups, cfg, seed):
    """Out-of-fold predictions: the regressor for group g is trained on all other groups."""
    u = np.zeros(len(Y))
    for g in np.unique(groups):
        m = groups == g
        u[m] = np.clip(make_regressor(cfg, seed).fit(X[~m], Y[~m]).predict(X[m]), 0, 1)
    return u


def select(u, tau):
    """Keep components with u >= tau; always keep the best one so no image is left empty."""
    keep = u >= tau
    if len(u) and not keep.any():
        keep[np.argmax(u)] = True
    return keep


def tau_grid(cfg):
    a, b, s = cfg["scorer"]["tau_grid"]
    return np.arange(a, b, s)
