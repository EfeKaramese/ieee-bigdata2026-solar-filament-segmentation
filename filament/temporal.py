"""Temporal-consistency distillation on unlabeled GONG bursts (exploratory experiment, not used by the final model).

A burst is a set of frames from one GONG station within a few minutes: filaments do not change on that time scale,
so frame-to-frame differences of a model's output are noise (seeing, clouds). Two uses:

* targets  — a teacher segments every frame; the mean over the burst's clean, aligned frames is a cleaner pseudo-label
             ("burst target") than any single-frame prediction ("single-frame target", classic self-training).
* students — trained on the labeled crops plus, at every step, two frames of one burst cropped at the same place:
             loss on the teacher target and (optionally) a consistency loss between the two frames.
Stability is measured on held-out bursts by scoring every pair of frames against each other with PQ (self-PQ).

Burst definition, cloud flags, alignment shifts and the held-out split come from the earlier burst-building step
(``kareler.csv`` / ``demetler.csv``); only the teacher changes here.
"""
import itertools
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import evaluate as ev
from . import postprocess as pp
from .data import read_cache
from .infer import predict
from .preprocess import preprocess_image


# ------------------------------------------------------------------------------------------------- burst tables
def load_bursts(gong_dir):
    """Frames table (stem, demet, temiz, dx, dy, hiza_ok, rol) and bursts table (demet, kullanildi, ayrik)."""
    KR = pd.read_csv(Path(gong_dir) / "kareler.csv")
    DM = pd.read_csv(Path(gong_dir) / "demetler.csv")
    return KR, DM


def shift(a, dx, dy):
    """Move the content of a by (-dx, -dy) whole pixels with zero fill (aligns a frame to the burst reference)."""
    sx, sy = -int(dx), -int(dy)
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    out[max(0, sy):H + min(0, sy), max(0, sx):W + min(0, sx)] = a[max(0, -sy):H + min(0, -sy), max(0, -sx):W + min(0, -sx)]
    return out


def _write(path, a):
    cv2.imwrite(str(path), a if a.dtype == np.uint8 else np.clip(np.round(a * 255), 0, 255).astype(np.uint8))


_JOB = {}


def _prep_frame(stem):
    full, pcfg, raw = _JOB["full"], _JOB["pcfg"], _JOB["raw"]
    img = cv2.imread(str(raw[stem]), cv2.IMREAD_GRAYSCALE)
    return stem, (None if img is None else preprocess_image(img, full, pcfg))


def build_targets(gong_dir, raw_dir, teacher, out_dir, cfg, device, workers=4, log=print):
    """Teacher targets for the training bursts (used, not held out):
    girdi/<frame>.png  aligned network input of every student frame
    tek/<frame>.png    teacher prediction of that frame alone (single-frame target)
    hedef/<burst>.png  mean teacher prediction over the burst's clean frames (burst target)
    The teacher sees each frame once, without flips (as in the original burst-building step)."""
    KR, DM = load_bursts(gong_dir)
    out = Path(out_dir)
    for k in ("girdi", "tek", "hedef"):
        (out / k).mkdir(parents=True, exist_ok=True)
    train_b = set(DM[DM.kullanildi & ~DM.ayrik].demet)
    need = KR[KR.demet.isin(train_b) & (KR.temiz | (KR.rol == "ogrenci"))]
    done = {p.stem for p in (out / "hedef").glob("*.png")}
    if done >= train_b:
        log(f"targets already built for all {len(train_b)} training bursts ({out})")
        return out
    raw = {}
    for ext in ("jpeg", "jpg", "png"):
        for p in Path(raw_dir).rglob(f"*.{ext}"):
            raw.setdefault(p.stem, p)
    if not raw and done:
        log(f"no raw frames under {raw_dir}: using the existing targets ({len(done)}/{len(train_b)} bursts)")
        return out
    missing = sorted(set(need.stem) - set(raw))
    log(f"raw frames found under {raw_dir}: {len(raw)} | needed: {len(need)} | missing: {len(missing)}")
    if len(missing) > 0.5 * len(need):
        ex = sorted(raw)[:3]
        raise FileNotFoundError(f"{len(missing)}/{len(need)} needed raw GONG frames are missing (e.g. {missing[:3]}); "
                                f"files found there look like {ex}. Check the raw GONG download (RAW_DIR).")
    if missing:
        log(f"WARNING: {len(missing)} raw frames not found (e.g. {missing[:3]}) — their bursts use fewer frames")
    todo = [b for b in sorted(train_b) if b not in done]
    log(f"{len(train_b)} training bursts ({len(done)} ready), {len(need)} frames needed")
    _JOB.update(full=cfg["data"]["full"], pcfg=cfg["preprocess"], raw=raw)
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(max(1, workers)) as pool:
        for i, b in enumerate(todo):
            fr = need[(need.demet == b) & need.stem.isin(raw)]
            xs = dict(pool.map(_prep_frame, fr.stem.tolist()))
            preds, clean = {}, []
            for r in fr.itertuples():
                x = xs.get(r.stem)
                if x is None:
                    continue
                xa = shift(x, r.dx, r.dy)
                pa = shift(predict([teacher], x, device, tta=False), r.dx, r.dy)
                preds[r.stem] = pa
                if r.temiz:
                    clean.append(pa)
                if r.rol == "ogrenci":
                    _write(out / "girdi" / f"{r.stem}.png", xa)
                    _write(out / "tek" / f"{r.stem}.png", pa)
            if clean:
                _write(out / "hedef" / f"{b}.png", np.mean(clean, 0))
            if (i + 1) % 25 == 0:
                log(f"  targets {len(done) + i + 1}/{len(train_b)} bursts ({(time.time() - t0) / 60:.1f} min)")
    _JOB.clear()
    log(f"targets ready ({(time.time() - t0) / 60:.1f} min)")
    return out


def student_bursts(gong_dir, target_dir):
    """Training bursts with a burst target and at least two student frames: {burst: [frame stems]}."""
    KR, DM = load_bursts(gong_dir)
    t = Path(target_dir)
    ok = {p.stem for p in (t / "hedef").glob("*.png")}
    have = {p.stem for p in (t / "girdi").glob("*.png")}
    g = KR[(KR.rol == "ogrenci") & KR.stem.isin(have) & KR.demet.isin(ok)].groupby("demet").stem.apply(list)
    return {b: s for b, s in g.items() if len(s) >= 2}


# ------------------------------------------------------------------------------------------------ pair dataset
class BurstPairs(Dataset):
    """Two frames of one burst, cropped at the same place with the same geometry (frames are aligned); photometric
    augmentation per frame. Returns x, target, valid, each (2, 1, K, K); valid = 0 where rotation padded."""

    def __init__(self, bursts, target_dir, target, cfg, length):
        self.b, self.keys, self.dir, self.target, self.cfg, self.n = bursts, list(bursts), Path(target_dir), target, cfg, length

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = np.random.default_rng(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        d, A = self.cfg["data"], self.cfg["augment"]
        full, K = d["full"], d["crop"]
        bid = self.keys[r.integers(len(self.keys))]
        sa, sb = (self.b[bid][j] for j in r.choice(len(self.b[bid]), 2, replace=False))
        xs = [read_cache(self.dir, "girdi", s, full) for s in (sa, sb)]
        if self.target == "burst":
            t = read_cache(self.dir, "hedef", bid, full)
            ts = [t, t]
        else:
            ts = [read_cache(self.dir, "tek", s, full) for s in (sa, sb)]
        ys, xs_ = np.nonzero(ts[0] > 0.3)
        if len(ys) and r.random() < d["filament_centered"]:
            k = r.integers(len(ys))
            cy, cx = ys[k] + r.integers(-K // 4, K // 4), xs_[k] + r.integers(-K // 4, K // 4)
        else:
            dy, dx = np.nonzero(xs[0] > 0)
            k = r.integers(len(dy)) if len(dy) else 0
            cy, cx = (dy[k], dx[k]) if len(dy) else (full // 2, full // 2)
        y0, x0 = int(np.clip(cy - K // 2, 0, full - K)), int(np.clip(cx - K // 2, 0, full - K))
        fh, fv = r.random() < 0.5, r.random() < 0.5
        M = (cv2.getRotationMatrix2D((K / 2, K / 2), r.uniform(-A["rotate_deg"], A["rotate_deg"]), r.uniform(*A["scale"]))
             if r.random() < A["p_affine"] else None)

        def geo(a):
            a = a[y0:y0 + K, x0:x0 + K]
            if fh: a = a[:, ::-1]
            if fv: a = a[::-1]
            a = np.ascontiguousarray(a)
            if M is not None:
                a = cv2.warpAffine(a, M, (K, K), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return a

        out_x = []
        for x in xs:
            x = geo(x)
            if r.random() < A["p_gamma"]:
                x = np.clip(x, 0, 1) ** r.uniform(*A["gamma"])
            out_x.append(x)
        out_t = [geo(t) for t in ts]
        v = geo(np.ones((full, full), np.float32))
        f = lambda L: torch.from_numpy(np.stack(L).astype(np.float32)[:, None])
        return f(out_x), f(out_t), f([v, v])


# ------------------------------------------------------------------------------------------------------- losses
def teacher_loss(logit, t, w, eps=1.0):
    """Soft BCE + soft Dice against the teacher target, masked by w."""
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * t * w).sum((1, 2, 3)) + eps) / ((p * w).sum((1, 2, 3)) + (t * w).sum((1, 2, 3)) + eps)
    return bce + dice.mean()


def consistency_loss(la, lb, valid):
    """Squared probability difference between two aligned frames, normalised by the predicted filament area."""
    pa, pb = torch.sigmoid(la), torch.sigmoid(lb)
    area = (0.5 * (pa + pb) * valid).sum().detach().clamp(min=1.0)
    return (((pa - pb) ** 2) * valid).sum() / area


def make_aux(bursts, target_dir, target, consistency, cfg, steps, ramp=(2, 7), w_u=1.0, w_tc=1.0, workers=2):
    """Auxiliary batch definition for ``train.train_model``."""
    ds = BurstPairs(bursts, target_dir, target, cfg, steps)
    r0, r1 = ramp

    def weight(ep):
        return 0.0 if ep < r0 else float(min(1.0, max(0.0, (ep - r0) / max(r1 - r0, 1))))

    def loss(seg_u, t, v):
        l_u = teacher_loss(seg_u, t, v)
        l_tc = consistency_loss(seg_u[0:1], seg_u[1:2], v[0:1] * v[1:2]) if consistency else seg_u.sum() * 0
        return w_u * l_u + w_tc * l_tc, dict(u=float(l_u.detach()), tc=float(l_tc.detach()))

    return dict(dataset=ds, loss=loss, weight=weight, workers=workers)


# ---------------------------------------------------------------------------------------------- stability
def label_iou(la, na, lb, nb):
    if na == 0 or nb == 0:
        return np.zeros((na, nb))
    c = np.bincount(la.astype(np.int64).ravel() * (nb + 1) + lb.astype(np.int64).ravel(),
                    minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    inter = c[1:, 1:].astype(float)
    return inter / np.maximum(c[1:, :].sum(1)[:, None] + c[:, 1:].sum(0)[None, :] - inter, 1)


def burst_stats(labs):
    """labs: [(label image, n, clean), ...] of one burst -> self-PQ parts (clean-clean, with cloudy), pair pixel IoUs,
    std and mean of the instance count."""
    p_tt, p_bi, piou = np.zeros(4), np.zeros(4), []
    for (la, na, ca), (lb, nb, cb) in itertools.combinations(labs, 2):
        part = np.array(ev.record_score(label_iou(la, na, lb, nb), np.ones(nb, bool)) if na else (0.0, 0, nb, 0))
        if ca and cb:
            p_tt += part
        else:
            p_bi += part
        ma, mb = la > 0, lb > 0
        piou.append(float((ma & mb).sum() / max((ma | mb).sum(), 1)))
    n = [k for _, k, _ in labs]
    return dict(tt=p_tt.tolist(), bi=p_bi.tolist(), piou=piou, std=float(np.std(n)), mean=float(np.mean(n)))


def held_out_bursts(gong_dir):
    KR, _ = load_bursts(gong_dir)
    g = KR[KR.rol == "ayrik"].groupby("demet").apply(lambda x: list(zip(x.stem, x.temiz.astype(bool)))).to_dict()
    return {b: k for b, k in g.items() if len(k) >= 2}


def stability(nets, gong_dir, cfg, device, done_path=None, log=print):
    """Self-PQ of each model on the held-out bursts (components without scorer, 4 flips); resumable per burst."""
    import json
    bursts = held_out_bursts(gong_dir)
    full = cfg["data"]["full"]
    p = cfg["postprocess"]
    done = json.loads(Path(done_path).read_text()) if done_path and Path(done_path).exists() else {}
    if done and set(next(iter(done.values()))) != set(nets):
        done = {}
    t0 = time.time()
    for b, frames in bursts.items():
        if b in done:
            continue
        labs = {n: [] for n in nets}
        for s, c in frames:
            x = read_cache(Path(gong_dir), "ayrik", s, full)
            for n, net in nets.items():
                labs[n].append((*pp.components(predict([net], x, device), p["t_lo"], p["t_hi"], p["close"], p["min_area"]), c))
        done[b] = {n: burst_stats(labs[n]) for n in nets}
        if done_path and (len(done) % 5 == 0 or len(done) == len(bursts)):
            tmp = Path(str(done_path) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, done_path)
            log(f"  stability {len(done)}/{len(bursts)} bursts ({(time.time() - t0) / 60:.1f} min)")
    if done_path:
        tmp = Path(str(done_path) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, done_path)
    order = [b for b in bursts if b in done]
    return {n: dict(tt=np.array([done[b][n]["tt"] for b in order]), bi=np.array([done[b][n]["bi"] for b in order]),
                    pixel_iou=float(np.mean([v for b in order for v in done[b][n]["piou"]])),
                    count_std=float(np.mean([done[b][n]["std"] for b in order])),
                    count_mean=float(np.mean([done[b][n]["mean"] for b in order]))) for n in nets}
