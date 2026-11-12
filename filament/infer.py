"""Inference: 4-flip test-time augmentation, averaging over models, optional reduced input resolution."""
import cv2
import numpy as np
import torch


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


@torch.no_grad()
def predict(nets, x, device, input_res=None, tta=True):
    """x: full x full float32 input -> full x full filament probability (mean over models and flips)."""
    full = x.shape[0]
    xi = x if not input_res or input_res == full else cv2.resize(x, (input_res, input_res), interpolation=cv2.INTER_AREA)
    dt = amp_dtype(device)
    flips = [(0, 0), (1, 0), (0, 1), (1, 1)] if tta else [(0, 0)]
    acc = np.zeros(xi.shape, np.float32)
    for net in nets:
        for fh, fv in flips:
            t = xi[:, ::-1] if fh else xi
            t = np.ascontiguousarray(t[::-1] if fv else t)
            with torch.autocast(device_type=device.type, enabled=dt is not None, dtype=dt or torch.float32):
                seg, _ = net(torch.from_numpy(t[None, None]).to(device))
            p = torch.sigmoid(seg.float())[0, 0].cpu().numpy()
            if fv:
                p = p[::-1]
            if fh:
                p = p[:, ::-1]
            acc += p
    acc /= len(nets) * len(flips)
    return acc if acc.shape[0] == full else cv2.resize(acc, (full, full), interpolation=cv2.INTER_LINEAR)
