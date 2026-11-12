"""High-level steps shared by the notebooks: out-of-fold probabilities, component candidates, scorer evaluation,
official evaluation and test inference."""
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import evaluate as ev
from . import postprocess as pp
from .data import read_cache
from .infer import predict
from .model import load_model
from .preprocess import preprocess_image
from .submission import image_rows


def log_to(path):
    """print + append to a log file (survives notebook disconnects)."""
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    return log


# ----------------------------------------------------------------------------------------- out-of-fold probabilities
def oof_probabilities(cfg, stems, fold_of, ckpts, cache_dir, out_dir, device, log=print):
    """Every training image predicted by the model that did not see it (full resolution, 4 flips), stored as uint8
    memmap (images x full x full). Completed folds are recorded so an interrupted run resumes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = cfg["data"]["full"]
    path, done_p = out / "oof_prob.npy", out / "oof_done.json"
    order_p = out / "oof_order.csv"
    if order_p.exists() and pd.read_csv(order_p).stem.tolist() != list(stems):
        raise RuntimeError(f"{order_p} has a different image order; delete {out} to recompute")
    pd.DataFrame(dict(stem=stems, fold=[fold_of[s] for s in stems])).to_csv(order_p, index=False)
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    if path.exists():
        O = np.load(path, mmap_mode="r+")
    else:
        O = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(stems), full, full))
    t0 = time.time()
    for f in sorted(set(fold_of[s] for s in stems)):
        if f in done:
            log(f"OOF fold {f}: ready")
            continue
        net = load_model(ckpts[f], cfg, device)
        for i, st in enumerate(stems):
            if fold_of[st] == f:
                p = predict([net], read_cache(cache_dir, "x", st), device, cfg["data"]["input_res"])
                O[i] = np.round(p * 255).astype(np.uint8)
        O.flush()
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        done.add(f)
        done_p.write_text(json.dumps(sorted(done)))
        log(f"OOF fold {f}: done ({(time.time() - t0) / 60:.1f} min)")
    return path


# ------------------------------------------------------------------------------------------- component candidates
_JOB = {}


def _candidate(i):
    J = _JOB
    st = J["stems"][i]
    if "O" not in J:
        J["O"] = np.load(J["prob_path"], mmap_mode="r")
    prob = J["O"][i].astype(np.float32) / 255.0
    cfg, full = J["cfg"], J["cfg"]["data"]["full"]
    x = read_cache(J["cache"], "x", st, full)
    lab, n = pp.components_cfg(prob, cfg)
    mats = [ev.iou_matrix(g, lab, n) for g in J["ann"].record_masks(st, full)]
    X19 = pp.base_features(lab, n, prob, x, st, full)
    X13 = pp.extra_features(lab, n, prob, x, full / 2048.0)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * (2048.0 / full) ** 2
    return st, dict(n=n, X19=X19, X13=X13, Y=pp.target_u(mats, n), mats=mats, area=area)


def candidates(prob_path, stems, ann, cache_dir, cfg, workers=4, log=print):
    """Components of every image with features, scorer target and per-record IoU matrices."""
    _JOB.clear()
    _JOB.update(prob_path=str(prob_path), stems=list(stems), ann=ann, cache=cache_dir, cfg=cfg)
    C, t0 = {}, time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for k, (st, c) in enumerate(pool.imap_unordered(_candidate, range(len(stems)), chunksize=2)):
                C[st] = c
                if (k + 1) % 100 == 0:
                    log(f"  candidates {k + 1}/{len(stems)} ({(time.time() - t0) / 60:.1f} min)")
    else:
        for i in range(len(stems)):
            st, c = _candidate(i)
            C[st] = c
    _JOB.clear()
    log(f"candidates: {sum(c['n'] for c in C.values())} components in {len(C)} images ({(time.time() - t0) / 60:.1f} min)")
    return C


# ---------------------------------------------------------------------------------------------- scorer evaluation
def design(C, stems, station=False, extra=True):
    X = np.concatenate([pp.feature_matrix(C[s]["X19"], C[s]["X13"] if extra else C[s]["X13"][:, :0], station)
                        for s in stems])
    Y = np.concatenate([C[s]["Y"] for s in stems])
    return X, Y


def parts(C, stems, U=None, tau=None):
    """Images x (sum IoU, TP, FP, FN); U=None keeps every component."""
    out = []
    for s in stems:
        keep = np.ones(C[s]["n"], bool) if U is None else pp.select(U[s], tau)
        out.append(ev.image_parts(C[s]["mats"], keep))
    return np.array(out, float)


def evaluate_scorer(C, stems, groups, cfg, seed, station=False, extra=True, oracle=False):
    """Cross-fitted scorer (by ``groups``), tau tuned on the pooled out-of-fold predictions."""
    X, Y = design(C, stems, station, extra)
    G = np.concatenate([np.full(C[s]["n"], groups[s]) for s in stems])
    u = Y.copy() if oracle else pp.crossfit(X, Y, G, cfg, seed)
    b = np.cumsum([0] + [C[s]["n"] for s in stems])
    U = {s: u[b[i]:b[i + 1]] for i, s in enumerate(stems)}
    curve = [(t, ev.pq(parts(C, stems, U, t))) for t in pp.tau_grid(cfg)]
    tau = max(curve, key=lambda z: z[1])[0]
    return dict(U=U, tau=float(tau), curve=curve, P=parts(C, stems, U, tau), P_all=parts(C, stems))


def fit_scorer(C, stems, cfg, seed, station=False):
    X, Y = design(C, stems, station)
    return pp.make_regressor(cfg, seed).fit(X, Y)


# -------------------------------------------------------------------------- prediction rows (official evaluation)
def _oof_rows(i):
    J = _JOB
    st = J["stems"][i]
    if "O" not in J:
        J["O"] = np.load(J["prob_path"], mmap_mode="r")
    prob = J["O"][i].astype(np.float32) / 255.0
    lab, n = pp.components_cfg(prob, J["cfg"])
    u = J["U"][st]
    assert len(u) == n, "component mismatch between candidate extraction and row export"
    return image_rows(st, lab, pp.select(u, J["tau"]), u, (2048, 2048))


def oof_frames(prob_path, stems, ann, U, tau, cfg, workers=4):
    """OOF predictions and ground truth in the official submission format."""
    _JOB.clear()
    _JOB.update(prob_path=str(prob_path), stems=list(stems), U=U, tau=tau, cfg=cfg)
    rows = []
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap(_oof_rows, range(len(stems)), chunksize=2):
                rows += r
    else:
        for i in range(len(stems)):
            rows += _oof_rows(i)
    _JOB.clear()
    gt = [r for s in stems for r in ann.gt_rows(s)]
    cols = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(gt, columns=cols)


# ----------------------------------------------------------------------------------------------------- test set
def predict_test(files, nets, scorer, tau, cfg, device, log=print):
    """Returns the submission DataFrame and {image stem: (height, width)} of the original images."""
    full = cfg["data"]["full"]
    rows, shapes, t0 = [], {}, time.time()
    for i, p in enumerate(files):
        st = Path(p).stem
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        shapes[st] = raw.shape[:2]
        x = preprocess_image(raw, full, cfg["preprocess"])
        prob = predict(nets, x, device, cfg["data"]["input_res"])
        lab, n = pp.components_cfg(prob, cfg)
        X = pp.feature_matrix(pp.base_features(lab, n, prob, x, st, full), pp.extra_features(lab, n, prob, x, full / 2048.0),
                              cfg["scorer"]["station_feature"])
        u = np.clip(scorer.predict(X), 0, 1) if n else np.zeros(0)
        rows += image_rows(st, lab, pp.select(u, tau), u, raw.shape[:2])
        if (i + 1) % 20 == 0:
            log(f"  test {i + 1}/{len(files)} ({(time.time() - t0) / 60:.1f} min)")
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]), shapes
