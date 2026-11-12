"""Analyses of the out-of-fold predictions, organised around the organisers' three challenges:

1. fine-structure fidelity       — performance by filament size, boundary complexity of matched pairs (barbs)
2. filament-background separation — error budget of misses and false positives, human-human agreement, scorer calibration
3. instance coherence             — fragmentation / over-merging counts, merge-oracle ceiling, closing-radius trade-off

Everything works on the candidate dictionary produced by ``pipeline.candidates`` (components, features, per-record IoU
matrices) plus one pass over the images for quantities that need the actual masks.
"""
import multiprocessing as mp
import time

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

from . import evaluate as ev
from . import postprocess as pp
from .evaluate import IOU_THR


# --------------------------------------------------------------------------------------------- shape descriptors
def shape_stats(m):
    """Area, isoperimetric complexity P^2 / (4 pi A) (1 for a disc, large for ragged / barbed outlines),
    skeleton end points and branch points of a binary mask (2048 px units assumed)."""
    area = int(m.sum())
    if area == 0:
        return dict(area=0, complexity=np.nan, ends=0, branches=0)
    edge = m & ~ndi.binary_erosion(m, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    per = float(edge.sum())
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    return dict(area=area, complexity=per ** 2 / (4 * np.pi * area), ends=int((sk & (nb == 1)).sum()),
                branches=int((sk & (nb >= 3)).sum()))


def set_iou(A, B):
    """IoU matrix between two lists of sorted flat-index arrays."""
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            inter = len(np.intersect1d(a, b, assume_unique=True))
            if inter:
                out[i, j] = inter / (len(a) + len(b) - inter)
    return out


# --------------------------------------------------------------------------------------------- one pass over images
_JOB = {}


def _image_pass(i):
    J = _JOB
    st = J["stems"][i]
    if "O" not in J:
        J["O"] = np.load(J["prob_path"], mmap_mode="r")
    cfg, full = J["cfg"], J["cfg"]["data"]["full"]
    prob = J["O"][i].astype(np.float32) / 255.0
    lab, n = pp.components_cfg(prob, cfg)
    c = J["C"][st]
    assert n == c["n"], f"{st}: components differ from the candidate file"
    keep = pp.select(J["U"][st], J["tau"])
    recs = J["ann"].record_masks(st, full)
    sc2 = (2048.0 / full) ** 2
    out = dict(gt_area=[np.array([len(g) for g in r], float) * sc2 for r in recs], fn=[], fp=[], pairs=[], human=[])
    # GT-to-GT IoU between records (annotator agreement) for multi-record images
    cross = {(a, b): set_iou(recs[a], recs[b]) for a in range(len(recs)) for b in range(len(recs)) if a != b}
    for r, m in enumerate(c["mats"]):
        if m.shape[0] == 0:
            continue
        mk = m[:, keep]
        hit_k = mk > IOU_THR
        others = [b for b in range(len(recs)) if b != r]
        # misses (FN) of this record
        for k in np.flatnonzero(~hit_k.any(1)):
            row = m[k]
            cat = ("removed_by_scorer" if (row > IOU_THR).any() else
                   "low_overlap" if (row > 0).any() else "not_detected")
            solo = None if not others else not any((cross[(r, b)][k] > IOU_THR).any() for b in others)
            out["fn"].append(dict(stem=st, record=r, area=out["gt_area"][r][k], category=cat, only_this_annotator=solo))
        # false positives of this record (kept components matching no GT of this record)
        kept_idx = np.flatnonzero(keep)
        for jj in np.flatnonzero(~hit_k.any(0)):
            j = kept_idx[jj]
            other_hit = any(c["mats"][b].shape[0] and (c["mats"][b][:, j] > IOU_THR).any() for b in others)
            cat = ("tp_for_other_annotator" if other_hit else
                   "partial_overlap" if (m[:, j] > 0).any() else "background")
            out["fp"].append(dict(stem=st, record=r, area=float(c["area"][j]), category=cat))
        # matched pairs: boundary complexity of GT vs prediction
        for k, jj in zip(*np.nonzero(hit_k)):
            j = kept_idx[jj]
            g = np.zeros(full * full, bool)
            g[recs[r][k]] = True
            g = g.reshape(full, full)
            p = lab == j + 1
            ys, xs = np.nonzero(g | p)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sg, sp = shape_stats(g[y0:y1, x0:x1]), shape_stats(p[y0:y1, x0:x1])
            out["pairs"].append(dict(stem=st, record=r, gt=int(k), pred=int(j), iou=float(m[k, j]),
                                     bbox=(int(y0), int(y1), int(x0), int(x1)),
                                     **{f"gt_{a}": v for a, v in sg.items()}, **{f"pred_{a}": v for a, v in sp.items()}))
    # human vs human: every ordered pair of records, one record's masks as "prediction" for the other
    for (a, b), iou in cross.items():
        n_gt, n_p = iou.shape
        if n_gt == 0 and n_p == 0:
            continue
        out["human"].append(ev.record_score(iou, np.ones(n_p, bool)))
    return st, out


def image_pass(prob_path, stems, ann, C, U, tau, cfg, workers=4, log=print):
    _JOB.clear()
    _JOB.update(prob_path=str(prob_path), stems=list(stems), ann=ann, C=C, U=U, tau=tau, cfg=cfg)
    res, t0 = {}, time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for k, (st, o) in enumerate(pool.imap_unordered(_image_pass, range(len(stems)), chunksize=2)):
                res[st] = o
                if (k + 1) % 100 == 0:
                    log(f"  image pass {k + 1}/{len(stems)} ({(time.time() - t0) / 60:.1f} min)")
    else:
        for i in range(len(stems)):
            st, o = _image_pass(i)
            res[st] = o
    _JOB.clear()
    log(f"image pass done ({(time.time() - t0) / 60:.1f} min)")
    return res


# ------------------------------------------------------------------------------------ challenge 1: size strata
def size_parts(C, stems, keep, gt_area, edges):
    """Images x bins x (sum IoU, TP, FP, FN). TP pairs and misses go to the GT's size bin, false positives to the
    prediction's size bin; bins add up exactly to the overall PQ."""
    nb = len(edges) + 1
    out = np.zeros((len(stems), nb, 4))
    for i, st in enumerate(stems):
        c, t = C[st], keep[st]
        pb = np.digitize(c["area"][t], edges)
        for m, ga in zip(c["mats"], gt_area[st]):
            m = m[:, t]
            gb = np.digitize(ga, edges)
            if m.shape[0] == 0:
                np.add.at(out[i, :, 2], pb, 1)
                continue
            if m.shape[1] == 0:
                np.add.at(out[i, :, 3], gb, 1)
                continue
            hit = m > IOU_THR
            np.add.at(out[i, :, 0], gb, (m * hit).sum(1))
            np.add.at(out[i, :, 1], gb, hit.sum(1))
            np.add.at(out[i, :, 3], gb, ~hit.any(1))
            np.add.at(out[i, :, 2], pb, ~hit.any(0))
    return out


# --------------------------------------------------------------------------------- challenge 2: calibration
def calibration(u, y, bins=10):
    """Reliability of the scorer: mean predicted u vs mean realised target per equal-count bin; ECE."""
    order = np.argsort(u)
    groups = np.array_split(order, bins)
    rows = [dict(u_mean=float(u[g].mean()), y_mean=float(y[g].mean()), n=len(g)) for g in groups if len(g)]
    ece = sum(r["n"] * abs(r["u_mean"] - r["y_mean"]) for r in rows) / max(len(u), 1)
    return rows, float(ece)


# ---------------------------------------------------------------------------------- challenge 3: coherence
def overlap_degrees(C, stems, keep):
    """GT overlapped by >= 2 kept predictions (fragmentation) and predictions overlapping >= 2 GT (over-merging),
    counted per record like the official one-to-many statistics."""
    frag = merge = 0
    for st in stems:
        for m in C[st]["mats"]:
            h = m[:, keep[st]] > 0
            if h.size:
                frag += int((h.sum(1) >= 2).sum())
                merge += int((h.sum(0) >= 2).sum())
    return dict(gt_fragmented=frag, pred_merged=merge)


def merge_oracle(C, stems, keep, gt_area):
    """Optimistic ceiling for fixing fragmentation: in every record, the kept fragments overlapping an unmatched GT
    (and not matched to another GT) are merged when their union would match it (IoU > 0.5). Decided per record
    with the ground truth, so it is an upper bound, not an achievable method."""
    parts, n_merged = [], 0
    for st in stems:
        c, t = C[st], keep[st]
        idx = np.flatnonzero(t)
        a = c["area"][idx]
        tot = np.zeros(4)
        for m, ga in zip(c["mats"], gt_area[st]):
            m = m[:, idx]
            s, tp, fp, fn = ev.record_score(m, np.ones(len(idx), bool))
            if m.shape[0] and m.shape[1]:
                hit = m > IOU_THR
                used = hit.any(0)
                for k in np.flatnonzero(~hit.any(1)):
                    S = np.flatnonzero((m[k] > 0) & ~used)
                    if len(S) < 2:
                        continue
                    inter = m[k, S] * (ga[k] + a[S]) / (1 + m[k, S])
                    iou = inter.sum() / (ga[k] + a[S].sum() - inter.sum())
                    if iou > IOU_THR:
                        used[S] = True
                        s, tp, fn, fp = s + iou, tp + 1, fn - 1, fp - len(S)
                        n_merged += 1
            tot += (s, tp, fp, fn)
        parts.append(tot)
    return np.array(parts), n_merged
