"""Submission rows (one RLE mask per filament, '<image>_<k>') and format checks."""
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu


def rle(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")


def image_rows(stem, lab, keep, u, out_hw=None):
    """Kept components in decreasing score order; label image resized (nearest) to the original size if needed."""
    if out_hw and lab.shape != tuple(out_hw):
        lab = cv2.resize(lab.astype(np.uint16), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    ids = np.flatnonzero(keep)
    rows, k = [], 0
    for j in ids[np.argsort(-u[ids])] + 1:
        m = lab == j
        if m.any():
            k += 1
            rows.append((f"{stem}_{k}", rle(m)))
    return rows


def check_submission(sub, shapes, n_check=None):
    """Unique ids, every image present, no empty masks, no overlapping masks within an image.
    shapes: {image stem: (height, width)} of the original images."""
    stems = list(shapes)
    assert sub.filament_id.is_unique, "duplicate filament ids"
    img = sub.filament_id.str.rsplit("_", n=1).str[0]
    missing = set(stems) - set(img)
    groups = list(sub.groupby(img))
    for st, g in groups[:n_check] if n_check else groups:
        M = np.stack([mu.decode({"size": list(shapes[st]), "counts": c.encode()}) for c in g.segmentation_rle])
        assert (M.reshape(len(M), -1).sum(1) > 0).all(), f"{st}: empty mask"
        assert M.sum(0).max() <= 1, f"{st}: overlapping masks"
    return dict(rows=len(sub), images=img.nunique(), missing_images=len(missing),
                per_image=float(len(sub) / max(img.nunique(), 1)))
