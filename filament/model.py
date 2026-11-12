"""ConvNeXt-tiny U-Net used for the final model.

* Encoder: timm ConvNeXt-tiny (ImageNet-22k pre-trained, fine-tuned on 1k), features at /4, /8, /16, /32.
* LayerNorm2d on every encoder output. timm's ``features_only`` skips the per-stage output norms of the official
  ConvNeXt segmentation code; without them the /16 and /32 activations reach |x| ~ 1e3 and overflow float16 after
  ~20 epochs. With them fp16 training is stable (and bf16 has the headroom anyway).
* ASPP on the /32 features without the image-level (global pooling) branch, and scSE whose channel gate uses a
  *local* 15x15 average: training sees 1024 crops, inference the full 2048 disk, so global statistics would differ.
* Two light stem branches at full and half resolution, because ConvNeXt downsamples by 4 in its first layer and thin
  filaments would otherwise only be reconstructed from /4 features.
* Two heads: filament mask and auxiliary spine (centre line).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import LayerNorm2d


def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))


class Up(nn.Module):
    def __init__(self, ci, cs, co):
        super().__init__()
        self.block = conv_block(ci + cs, co)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1) if skip is not None else x)


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-excitation (Roy et al. 2018) with a local channel gate."""

    def __init__(self, c, r=8, k=15):
        super().__init__()
        m = max(c // r, 8)
        self.k = k
        self.c = nn.Sequential(nn.Conv2d(c, m, 1), nn.ReLU(True), nn.Conv2d(m, c, 1), nn.Sigmoid())
        self.s = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        local = F.avg_pool2d(x, self.k, stride=1, padding=self.k // 2, count_include_pad=False)
        return x * self.c(local) + x * self.s(x)


class UpSE(Up):
    def __init__(self, ci, cs, co, k=15):
        super().__init__(ci, cs, co)
        self.se = SCSE(co, k=k)

    def forward(self, x, skip=None):
        return self.se(super().forward(x, skip))


class ASPP(nn.Module):
    """DeepLabv3 atrous spatial pyramid pooling without the global pooling branch."""

    def __init__(self, ci, co=256, rates=(6, 12, 18)):
        super().__init__()
        branch = lambda k, d: nn.Sequential(nn.Conv2d(ci, co, k, padding=0 if k == 1 else d, dilation=d, bias=False),
                                            nn.BatchNorm2d(co), nn.ReLU(True))
        self.branches = nn.ModuleList([branch(1, 1)] + [branch(3, d) for d in rates])
        self.fuse = nn.Sequential(nn.Conv2d(co * (1 + len(rates)), co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], 1))


class ConvNeXtUNet(nn.Module):
    def __init__(self, encoder="convnext_tiny.fb_in22k_ft_in1k", pretrained=False, out_norm=True,
                 aspp_rates=(6, 12, 18), scse_kernel=15):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        c1, c2, c3, c4 = self.enc.feature_info.channels()
        self.out_norm = out_norm
        if out_norm:
            self.norms = nn.ModuleList([LayerNorm2d(c) for c in (c1, c2, c3, c4)])
        self.aspp = ASPP(c4, 256, tuple(aspp_rates))
        k = scse_kernel
        self.u3, self.u2, self.u1 = UpSE(256, c3, 256, k), UpSE(256, c2, 128, k), UpSE(128, c1, 64, k)
        self.stem_full = conv_block(3, 16)
        self.stem_half = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                                       nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.u0, self.uf = UpSE(64, 32, 32, k), Up(32, 16, 16)
        self.head_seg, self.head_spine = nn.Conv2d(16, 1, 1), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Set to True by the training guard after repeated non-finite losses: the encoder then runs in float32.
        self.register_buffer("enc_fp32", torch.tensor(False))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        xn = (x - self.mean) / self.std
        if bool(self.enc_fp32):
            with torch.autocast(device_type=x.device.type, enabled=False):
                feats = self.enc(xn.float())
        else:
            feats = self.enc(xn)
        if self.out_norm:
            feats = [n(f) for n, f in zip(self.norms, feats)]
        e1, e2, e3, e4 = feats
        k1 = self.stem_full(x)
        k2 = self.stem_half(k1)
        d = self.u1(self.u2(self.u3(self.aspp(e4), e3), e2), e1)
        d = self.uf(self.u0(d, k2), k1)
        return self.head_seg(d), self.head_spine(d)


def build_model(cfg, pretrained=False):
    m = cfg["model"]
    return ConvNeXtUNet(m["encoder"], pretrained, m["encoder_out_norm"], m["aspp_rates"], m["scse_kernel"])


def load_model(path, cfg, device="cpu"):
    """Load a checkpoint (full or half precision state dict) into a float32 model in eval mode."""
    net = build_model(cfg, pretrained=False)
    sd = torch.load(path, map_location="cpu")
    sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def export_half(src, dst):
    """Half-precision copy of a state dict for the repository (< 100 MB per file)."""
    sd = torch.load(src, map_location="cpu")
    torch.save({k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}, dst)
