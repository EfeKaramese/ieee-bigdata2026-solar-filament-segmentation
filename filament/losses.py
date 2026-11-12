"""Training loss: BCE + Dice on the filament mask, clDice (ramped in), BCE on the auxiliary spine head."""
import torch
import torch.nn.functional as F


def soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)), -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))


def soft_open(x):
    return F.max_pool2d(soft_erode(x), (3, 3), (1, 1), (1, 1))


def soft_skel(x, iters):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        d = F.relu(x - soft_open(x))
        skel = skel + F.relu(d - skel * d)
    return skel


def dice_loss(logit, t, eps=1.0):
    p = torch.sigmoid(logit)
    return (1 - (2 * (p * t).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()


def cldice_loss(logit, t, iters, eps=1.0):
    p = torch.sigmoid(logit)
    sp, st = soft_skel(p, iters), soft_skel(t, iters)
    prec = ((sp * t).sum() + eps) / (sp.sum() + eps)
    sens = ((st * p).sum() + eps) / (st.sum() + eps)
    return 1 - 2 * prec * sens / (prec + sens)


def cldice_scale(epoch, cfg):
    a, b = cfg["loss"]["cldice_ramp"]
    if cfg["train"]["epochs"] <= b - a:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - a) / (b - a))))


def total_loss(seg, spine, y, s, w, scale, cfg):
    L = cfg["loss"]
    bce = (F.binary_cross_entropy_with_logits(seg, y, reduction="none") * w).mean()
    top = cldice_loss(seg, y, L["cldice_iters"]) if scale > 0 and L["w_cldice"] > 0 else seg.sum() * 0
    return (L["w_bce"] * bce + L["w_dice"] * dice_loss(seg, y) + L["w_cldice"] * scale * top
            + L["w_spine"] * F.binary_cross_entropy_with_logits(spine, s))
