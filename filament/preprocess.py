"""Image preprocessing: solar disk detection, radial limb-darkening flattening, contrast enhancement.

Output is a single-channel float32 image in [0, 1] (zero outside the disk), the only input the network sees.
"""
import numpy as np
import cv2


def _refine_radius(img, cy, cx, r, span=0.04, max_shift=0.03, nbins=120):
    """Refine the Hough radius to the steepest intensity drop of the radial median profile (the limb)."""
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    edges = np.linspace((1 - span) * r, (1 + span) * r, nbins + 1)
    idx = np.digitize(rr, edges) - 1
    prof = np.full(nbins, np.nan, np.float32)
    for b in range(nbins):
        m = idx == b
        if m.sum() > 100:
            prof[b] = np.median(img[m])
    good = ~np.isnan(prof)
    if good.sum() < 10:
        return r
    p = np.convolve(np.interp(np.arange(nbins), np.flatnonzero(good), prof[good]), np.ones(5) / 5, mode="same")
    k = int(np.argmin(np.gradient(p)))
    rn = 0.5 * (edges[k] + edges[k + 1])
    return r if abs(rn - r) > max_shift * r else float(rn)


def find_disk(img, radius_prior=905 / 2048, radius_tol=0.03):
    """Solar disk centre and radius (cy, cx, r) in pixels via a Hough circle on a 512 px copy, then limb refinement."""
    H, W = img.shape
    sc = 512.0 / max(H, W)
    sm = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
    sm = (sm - sm.min()) / max(float(sm.max() - sm.min()), 1e-6)
    u8 = cv2.medianBlur((sm * 255).astype(np.uint8), 5)
    h, w = u8.shape
    rp = radius_prior * max(h, w)
    for p2 in (30, 22, 15):
        c = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp=1.0, minDist=h, param1=100, param2=p2,
                             minRadius=int((1 - radius_tol) * rp), maxRadius=int((1 + radius_tol) * rp))
        if c is not None:
            cx, cy, r = c[0][0]
            cy, cx, r = cy / sc, cx / sc, r / sc
            if abs(cy - H / 2) < 0.15 * H and abs(cx - W / 2) < 0.15 * W:
                return float(cy), float(cx), float(_refine_radius(img, cy, cx, r))
    return H / 2, W / 2, radius_prior * max(H, W)


def flatten(img, cy, cx, r, nbins=64, r_frac=0.97, clip=(0.75, 1.25)):
    """Divide by the radial median profile (limb darkening), clip the ratio and rescale to [0, 1] inside the disk."""
    s = img.astype(np.float32)
    h, w = s.shape
    sc = 512.0 / max(h, w)
    sm = cv2.resize(s, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    Ys, Xs = np.ogrid[:sm.shape[0], :sm.shape[1]]
    rrs = np.sqrt((Ys - cy * sc) ** 2 + (Xs - cx * sc) ** 2)
    in_disk = rrs < r_frac * r * sc
    idx = np.clip(np.digitize(rrs, np.linspace(0, r_frac * r * sc, nbins + 1)) - 1, 0, nbins - 1)
    prof = np.full(nbins, np.nan, np.float32)
    for b in range(nbins):
        m = (idx == b) & in_disk
        if m.sum() > 20:
            prof[b] = np.median(sm[m])
    good = ~np.isnan(prof)
    prof = np.maximum(np.interp(np.arange(nbins), np.flatnonzero(good), prof[good]), 1e-3)
    Y, X = np.ogrid[:h, :w]
    rr = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    disk = rr < r_frac * r
    e = np.linspace(0, r_frac * r, nbins + 1)
    bg = np.interp(rr.ravel(), (e[:-1] + e[1:]) / 2, prof).reshape(h, w)
    ratio = np.clip(s / np.maximum(bg, 1e-3), clip[0], clip[1])
    return np.where(disk, (ratio - clip[0]) / (clip[1] - clip[0]), 0.0).astype(np.float32), disk


def enhance(flat, disk, clip_limit=2.5, tile=8, blur_sigma=0.7):
    """Light Gaussian denoising followed by CLAHE; zero outside the disk."""
    u8 = np.clip(cv2.GaussianBlur(flat, (3, 3), blur_sigma) * 255, 0, 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(u8).astype(np.float32) / 255.0
    out[~disk] = 0.0
    return out


def preprocess_image(img, size, pcfg):
    """Raw grey image (any size) -> size x size network input in [0, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cy, cx, r = find_disk(img, pcfg["disk_radius_prior"], pcfg["disk_radius_tol"])
    flat, disk = flatten(img, cy, cx, r, nbins=pcfg["flat_bins"], clip=tuple(pcfg["flat_clip"]))
    return enhance(flat, disk, pcfg["clahe_clip"], pcfg["clahe_tile"], pcfg["blur_sigma"])
