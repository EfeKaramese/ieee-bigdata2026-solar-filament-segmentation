"""Annotations, multi-annotator targets, folds, the preprocessing cache and the training dataset.

Annotation structure (MAGFiLO / Kaggle JSON): several *records* (annotators) can label the same image; every record is a
separate entry in ``images`` with id ``"<annotator>-<image>"``. The official metric scores every record separately and
pools the counts, so all ground-truth handling here is per record.
"""
import collections
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu
from torch.utils.data import Dataset

from .preprocess import preprocess_image

IMAGE_EXT = (".jpeg", ".jpg", ".png")


class Annotations:
    def __init__(self, json_path, image_dir=None):
        d = json.load(open(json_path))
        self.by_img = collections.defaultdict(list)
        for a in d["annotations"]:
            self.by_img[a["image_id"]].append(a)
        self.by_file = collections.defaultdict(list)
        for im in d["images"]:
            self.by_file[im["file_name"]].append(im["id"])
        self.stem2file = {Path(f).stem: f for f in self.by_file}
        paths = {Path(p).stem: p for p in glob.glob(f"{image_dir}/*")} if image_dir else {}
        rows = []
        for f, ids in sorted(self.by_file.items()):
            st = Path(f).stem
            rows.append(dict(stem=st, file_name=f, path=paths.get(st), station=st[-2], K=len(ids),
                             n_fil=float(np.mean([len(self.by_img[i]) for i in ids]))))
        self.meta = pd.DataFrame(rows)
        if image_dir:
            self.meta = self.meta[self.meta.path.notna()].reset_index(drop=True)

    def records(self, stem):
        return self.by_file[self.stem2file[stem]]

    def record_masks(self, stem, size=2048):
        """[[flat pixel indices of each filament] for each record] at size x size."""
        sc = size / 2048.0
        out = []
        for iid in self.records(stem):
            ms = []
            for a in self.by_img[iid]:
                segs = [list(np.asarray(p, np.float64) * sc) for p in a["segmentation"]]
                m = mu.decode(mu.merge(mu.frPyObjects(segs, size, size))).astype(bool)
                if m.any():
                    ms.append(np.flatnonzero(m.ravel()).astype(np.int32))
            out.append(ms)
        return out

    def targets(self, stem, size=2048, spine_px=5):
        """Agreement map (fraction of records marking each pixel; its mean-over-records soft target) and spine map."""
        ids = self.records(stem)
        sc = size / 2048.0
        acc = np.zeros((size, size), np.float32)
        spine = np.zeros((size, size), np.uint8)
        for iid in ids:
            m = np.zeros((size, size), np.uint8)
            for a in self.by_img[iid]:
                segs = [list(np.asarray(p, np.float64) * sc) for p in a["segmentation"]]
                m |= mu.decode(mu.merge(mu.frPyObjects(segs, size, size)))
                if a.get("spine") and len(a["spine"]) >= 4:
                    p = (np.array(a["spine"], np.float32).reshape(-1, 2) * sc).round().astype(np.int32)
                    cv2.polylines(spine, [p], False, 1, max(1, int(round(spine_px * sc))))
            acc += m
        return np.clip(acc / max(len(ids), 1), 0, 1), spine.astype(np.float32)

    def gt_rows(self, stem):
        """Ground truth in the official submission format: one row per filament, id '<record>_<k>'."""
        rows = []
        for iid in self.records(stem):
            for k, a in enumerate(self.by_img[iid], 1):
                m = mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048)))
                if m.any():
                    rows.append((f"{iid}_{k}", mu.encode(np.asfortranarray(m))["counts"].decode("ascii")))
        return rows


def make_folds(meta, n_folds=4, seed=42):
    """Station-stratified folds, balanced in filament count within each station."""
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    meta["fold"] = -1
    for _, g in meta.groupby("station"):
        order = g.assign(r=rng.random(len(g))).sort_values(["n_fil", "r"]).index
        for i, ix in enumerate(order):
            meta.at[ix, "fold"] = i % n_folds
    return meta[["stem", "fold"]]


# ----------------------------------------------------------------------------------------------- preprocessing cache
CACHE_KINDS = ("x", "y", "s")      # x: network input, y: annotator agreement (mean target), s: spine
_JOB = {}


def _cache_one(stem):
    ann, cache, size, cfg = _JOB["ann"], _JOB["cache"], _JOB["size"], _JOB["cfg"]
    if all((cache / k / f"{stem}.png").exists() for k in CACHE_KINDS):
        return stem
    path = ann.meta.set_index("stem").at[stem, "path"]
    x = preprocess_image(cv2.imread(path, cv2.IMREAD_GRAYSCALE), size, cfg["preprocess"])
    y, s = ann.targets(stem, size, cfg["data"]["spine_px"])
    for arr, k in ((x, "x"), (y, "y"), (s, "s")):
        cv2.imwrite(str(cache / k / f"{stem}.png"), np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8))
    return stem


def build_cache(ann, stems, cache_dir, cfg, workers=4, log=print):
    """Preprocess every training image once (input, agreement target, spine target) as 8-bit PNGs."""
    cache = Path(cache_dir)
    for k in CACHE_KINDS:
        (cache / k).mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not all((cache / k / f"{s}.png").exists() for k in CACHE_KINDS)]
    log(f"cache {cache}: {len(stems) - len(todo)} ready, {len(todo)} to build")
    if not todo:
        return cache
    _JOB.update(ann=ann, cache=cache, size=cfg["data"]["full"], cfg=cfg)
    t0 = time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_cache_one, todo, chunksize=4)):
                if (i + 1) % 100 == 0:
                    log(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    else:
        for s in todo:
            _cache_one(s)
    log(f"cache built in {time.time() - t0:.0f}s")
    return cache


def read_cache(cache_dir, kind, stem, size=None):
    a = cv2.imread(str(Path(cache_dir) / kind / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if size and a.shape[0] != size:
        a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    return a


def make_target(agreement, cfg):
    """Training target and per-pixel BCE weight from the agreement map.

    mean : y = agreement (soft target, expected per-record label), weight 1.
    union: y = 1 where any annotator marked the pixel; inside the union the BCE weight grows with agreement
           (floor + (1 - floor) * agreement), outside it is 1.
    """
    if cfg["data"]["target"] == "mean":
        return agreement, np.ones_like(agreement)
    if cfg["data"]["target"] == "union":
        u = (agreement > 1e-3).astype(np.float32)
        f = cfg["data"]["union_weight_floor"]
        return u, np.where(u > 0, f + (1 - f) * agreement, 1.0).astype(np.float32)
    raise ValueError(cfg["data"]["target"])


def binary_target(agreement, cfg):
    """Binary mask used for validation IoU / checkpoint selection, consistent with the training target."""
    return agreement > (1e-3 if cfg["data"]["target"] == "union" else 0.5)


class CropDataset(Dataset):
    """Training samples: input_res == full -> random crops (half centred on filaments); input_res < full -> whole
    disk downscaled. Augmentations: flips, rotation, scale, gamma."""

    def __init__(self, stems, cache_dir, cfg):
        self.cache, self.cfg = cache_dir, cfg
        d = cfg["data"]
        self.full, self.res = d["full"], d["input_res"]
        self.crop = d["crop"] if self.res == self.full else self.res
        self.stems = [s for s in stems for _ in range(d["crops_per_image"] if self.res == self.full else 1)]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        st = self.stems[i]
        r = np.random.default_rng(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        a, A = self.cfg["data"], self.cfg["augment"]
        size = self.res
        x = read_cache(self.cache, "x", st, size)
        agr = read_cache(self.cache, "y", st, size)
        s = read_cache(self.cache, "s", st, size)
        y, w = make_target(agr, self.cfg)
        K = self.crop
        if K < size:
            ys, xs = np.nonzero(agr > 0.3)
            if len(ys) and r.random() < a["filament_centered"]:
                k = r.integers(len(ys))
                cy, cx = ys[k] + r.integers(-K // 4, K // 4), xs[k] + r.integers(-K // 4, K // 4)
            else:
                dy, dx = np.nonzero(x > 0)
                k = r.integers(len(dy)) if len(dy) else 0
                cy, cx = (dy[k], dx[k]) if len(dy) else (size // 2, size // 2)
            y0 = int(np.clip(cy - K // 2, 0, size - K))
            x0 = int(np.clip(cx - K // 2, 0, size - K))
            x, y, s, w = (t[y0:y0 + K, x0:x0 + K] for t in (x, y, s, w))
        if r.random() < 0.5:
            x, y, s, w = (t[:, ::-1] for t in (x, y, s, w))
        if r.random() < 0.5:
            x, y, s, w = (t[::-1] for t in (x, y, s, w))
        x, y, s, w = (np.ascontiguousarray(t) for t in (x, y, s, w))
        if r.random() < A["p_affine"]:
            M = cv2.getRotationMatrix2D((K / 2, K / 2), r.uniform(-A["rotate_deg"], A["rotate_deg"]), r.uniform(*A["scale"]))
            kw = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            x, y, s = (cv2.warpAffine(t, M, (K, K), borderValue=0, **kw) for t in (x, y, s))
            w = cv2.warpAffine(w, M, (K, K), borderValue=1, **kw)
        if r.random() < A["p_gamma"]:
            x = np.clip(x, 0, 1) ** r.uniform(*A["gamma"])
        f = lambda t: torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)[None])
        return f(x), f(y), f(s), f(w)


def find_files(root, pattern):
    return sorted(glob.glob(os.path.join(str(root), "**", pattern), recursive=True))
