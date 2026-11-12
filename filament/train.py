"""Training loop.

* Mixed precision: bf16 on GPUs with compute capability >= 8 (A100/L4), fp16 + GradScaler otherwise (T4/P100).
* Non-finite guard: a batch with a non-finite loss is skipped; after ``nan_patience`` consecutive skipped batches, or if
  any weight becomes non-finite, the model rolls back to the last completed epoch and the encoder switches to float32.
* Resumable: the full state (model, optimiser, scheduler, scaler, history) is written atomically after every epoch, so
  a run cut by a session limit continues from the last epoch. Every epoch uses its own seed, so a resumed run sees the
  same sample order as an uninterrupted one.
* Checkpoint selection: best validation IoU (every ``val_every`` epochs) at full resolution.
"""
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CropDataset, binary_target, find_files, read_cache
from .infer import amp_dtype, predict
from .losses import cldice_scale, total_loss
from .model import build_model


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def val_iou(net, stems, cache_dir, cfg, device):
    net.eval()
    inter = union = 0
    for st in stems:
        p = predict([net], read_cache(cache_dir, "x", st), device, cfg["data"]["input_res"], tta=False) > 0.5
        t = binary_target(read_cache(cache_dir, "y", st), cfg)
        inter += (p & t).sum()
        union += (p | t).sum()
    return float(inter / max(union, 1))


def _restore_previous(tag, work, resume_dirs, log):
    """Copy a finished model or an interrupted state from an earlier session's output (e.g. Kaggle input datasets)."""
    ckpt, done, state = work / "ckpt" / f"{tag}.pt", work / "ckpt" / f"{tag}.done.json", work / "state" / f"{tag}_last.pt"
    if done.exists() or state.exists():
        return
    for root in resume_dirs:
        d, c = find_files(root, f"{tag}.done.json"), find_files(root, f"{tag}.pt")
        if d and c:
            shutil.copy(d[0], done); shutil.copy(c[0], ckpt)
            log(f"[{tag}] finished model found in {root}")
            return
    for root in resume_dirs:
        s = find_files(root, f"{tag}_last.pt")
        if s:
            shutil.copy(s[0], state)
            c = find_files(root, f"{tag}.pt")
            if c:
                shutil.copy(c[0], ckpt)
            log(f"[{tag}] interrupted state found in {root}")
            return


def train_model(cfg, tag, train_stems, val_stems, cache_dir, work_dir, device, seed, pretrained=True,
                deadline=None, resume_dirs=(), log=print, aux=None):
    """Train one model. Returns dict(status="done"|"paused", ckpt=Path, history=list, best=float).

    aux (optional, e.g. ``temporal.make_aux``): an extra unlabeled sample per step, concatenated to the labeled batch:
    dict(dataset -> (x, t, v) each (n, 1, K, K); loss(seg_u, t, v) -> (loss, parts); weight(epoch) -> float; workers)."""
    work = Path(work_dir)
    (work / "ckpt").mkdir(parents=True, exist_ok=True)
    (work / "state").mkdir(parents=True, exist_ok=True)
    ckpt, done, state_p = work / "ckpt" / f"{tag}.pt", work / "ckpt" / f"{tag}.done.json", work / "state" / f"{tag}_last.pt"
    _restore_previous(tag, work, resume_dirs, log)
    if done.exists():
        info = json.loads(done.read_text())
        log(f"[{tag}] already trained (best val IoU {info['best']:.4f})")
        return dict(status="done", ckpt=ckpt, history=info["history"], best=info["best"])

    T = cfg["train"]
    E, bs = T["epochs"], T["batch_size"]
    ds = CropDataset(train_stems, cache_dir, cfg)
    steps = len(ds) // bs
    dt = amp_dtype(device)
    vstems = list(val_stems)[::T["val_stride"]]

    def fresh():
        torch.manual_seed(seed)
        net = build_model(cfg, pretrained and not T.get("init_checkpoint")).to(device)
        if T.get("init_checkpoint"):
            sd = torch.load(T["init_checkpoint"], map_location="cpu")
            net.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()})
        enc = [p for n, p in net.named_parameters() if n.startswith("enc.")]
        rest = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
        groups = [dict(params=enc, lr=T["lr"] * T["encoder_lr_mult"]), dict(params=rest, lr=T["lr"])]
        opt = torch.optim.AdamW(groups, lr=T["lr"], weight_decay=T["weight_decay"])
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], epochs=E,
                                                  steps_per_epoch=steps, pct_start=T["pct_start"])
        scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
        return net, opt, sch, scaler

    def load_state(net, opt, sch, scaler):
        s = torch.load(state_p, map_location="cpu", weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sch.load_state_dict(s["sch"])
        scaler.load_state_dict(s["scaler"])
        return s["epoch"] + 1, s["best"], s["history"]

    net, opt, sch, scaler = fresh()
    ep, best, history = 0, -1.0, []
    if state_p.exists():
        ep, best, history = load_state(net, opt, sch, scaler)
        log(f"[{tag}] resuming at epoch {ep} (best val IoU so far {best:.4f}, encoder fp32={bool(net.enc_fp32)})")
    log(f"[{tag}] {len(train_stems)} train / {len(val_stems)} val images, {steps} steps/epoch, "
        f"precision {dt if dt else 'fp32'}")

    t_epoch, nan_streak, rollbacks = None, 0, 0
    while ep < E:
        if deadline and t_epoch and time.time() + 1.15 * t_epoch > deadline:
            log(f"[{tag}] time budget reached before epoch {ep}; state saved, run again to continue")
            return dict(status="paused", ckpt=ckpt, history=history, best=best)
        t0 = time.time()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        torch.manual_seed(seed * 1000 + ep)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=T["workers"],
                        pin_memory=device.type == "cuda", drop_last=True)
        if aux:
            gu = torch.Generator().manual_seed(seed * 1000 + ep + 500)
            dlu = DataLoader(aux["dataset"], batch_size=1, shuffle=False, generator=gu, num_workers=aux["workers"],
                             pin_memory=device.type == "cuda")
            lam, aux_parts = aux["weight"](ep), {}
        net.train()
        scale, losses, rollback = cldice_scale(ep, cfg), [], False
        for batch in (zip(dl, dlu) if aux else ((b, None) for b in dl)):
            (x, y, s, w), ub = batch
            x, y, s, w = (t.to(device, non_blocking=True) for t in (x, y, s, w))
            nb = x.shape[0]
            if ub is not None:
                xu, tu, vu = (t[0].to(device, non_blocking=True) for t in ub)
                x_in = torch.cat([x, xu])
            else:
                x_in = x
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=dt is not None, dtype=dt or torch.float32):
                seg, spine = net(x_in)
            loss = total_loss(seg[:nb].float(), spine[:nb].float(), y, s, w, scale, cfg)
            if ub is not None:
                l_aux, parts = aux["loss"](seg[nb:].float(), tu, vu)
                loss = loss + lam * l_aux
                for k, v in parts.items():
                    aux_parts.setdefault(k, []).append(v)
            if not torch.isfinite(loss):
                nan_streak += 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sch.step()                       # keep the LR schedule tied to the epoch
                if nan_streak >= T["nan_patience"]:
                    rollback = True
                    break
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), T["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sch.step()
            losses.append(loss.item())
        finite = all(torch.isfinite(v).all() for v in net.state_dict().values() if torch.is_floating_point(v))
        if rollback or not finite:
            rollbacks += 1
            if rollbacks > 3:
                raise RuntimeError(f"[{tag}] training keeps diverging")
            log(f"[{tag}] epoch {ep}: non-finite {'loss' if rollback else 'weights'} -> roll back, encoder to fp32")
            net, opt, sch, scaler = fresh()
            if state_p.exists():
                ep, best, history = load_state(net, opt, sch, scaler)
            else:
                ep, best, history = 0, -1.0, []
            net.enc_fp32.fill_(True)
            nan_streak = 0
            continue
        rec = dict(epoch=ep, loss=float(np.mean(losses)) if losses else float("nan"), skipped=steps - len(losses))
        if aux:
            rec.update(aux_weight=lam, **{f"aux_{k}": float(np.mean(v)) for k, v in aux_parts.items()})
        if (ep + 1) % T["val_every"] == 0 or ep == E - 1:
            rec["val_iou"] = val_iou(net, vstems, cache_dir, cfg, device)
            if rec["val_iou"] > best:
                best = rec["val_iou"]
                _atomic_save(net.state_dict(), ckpt)
        rec["minutes"] = (time.time() - t0) / 60
        history.append(rec)
        _atomic_save(dict(model=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(),
                          epoch=ep, best=best, history=history), state_p)
        log(f"[{tag}] ep{ep:02d} loss={rec['loss']:.4f}"
            + (f" (λ={rec['aux_weight']:.2f} u={rec.get('aux_u', 0):.3f} tc={rec.get('aux_tc', 0):.4f})" if aux else "")
            + (f" val_IoU={rec['val_iou']:.4f} (best {best:.4f})"
            if "val_iou" in rec else "") + f" | {rec['minutes']:.1f} min")
        t_epoch = time.time() - t0
        ep += 1
    done.write_text(json.dumps(dict(best=best, history=history, encoder_fp32=bool(net.enc_fp32)), indent=1))
    state_p.unlink(missing_ok=True)
    return dict(status="done", ckpt=ckpt, history=history, best=best)
