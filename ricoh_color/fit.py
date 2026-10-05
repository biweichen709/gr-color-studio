"""Fit a 3D LUT from pixel-aligned image pairs (source colour -> target colour).

The LUT is identity plus a correction field. Samples are splatted onto the
grid with trilinear weights; a smoothness term fills sparsely observed nodes
and a weak pull keeps unobserved colours close to "no change".
"""

import numpy as np

from .lut import apply, cell, corners, identity


def _splat(n, src, values):
    i0, f = cell(n, src)
    size = n**3
    num = np.zeros((size, 3))
    den = np.zeros(size)
    for (ri, gi, bi), w in corners(i0, f):
        index = ((ri * n + gi) * n + bi).ravel()
        w = w.ravel()
        den += np.bincount(index, w, minlength=size)
        for c in range(3):
            num[:, c] += np.bincount(index, w * values[:, c], minlength=size)
    return num.reshape(n, n, n, 3), den.reshape(n, n, n)


def _laplacian(x):
    out = np.zeros_like(x)
    for axis in range(3):
        step = np.diff(x, axis=axis)
        lower = [slice(None)] * x.ndim
        upper = [slice(None)] * x.ndim
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        out[tuple(lower)] -= step
        out[tuple(upper)] += step
    return out


def _solve(num, den, lam, eps, iterations=400, tol=1e-9):
    """Conjugate gradients for (diag(den) + eps + lam * Laplacian) x = num."""
    diag = (den + eps)[..., None]

    def operator(x):
        return diag * x + lam * _laplacian(x)

    x = np.zeros_like(num)
    r = num.copy()
    p = r.copy()
    rs = float((r * r).sum())
    limit = tol * max(float((num * num).sum()), 1e-30)
    for _ in range(iterations):
        if rs <= limit:
            break
        ap = operator(p)
        alpha = rs / float((p * ap).sum())
        x += alpha * p
        r -= alpha * ap
        rs, previous = float((r * r).sum()), rs
        p = r + (rs / previous) * p
    return x


def fit_lut(src, dst, grid=33, smooth=1.0, passes=3, max_samples=400_000, seed=0):
    """Return (lut, info) mapping src colours (K, 3) to dst colours (K, 3), both 0..1."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("source and target must have the same number of pixels")
    if len(src) > max_samples:
        keep = np.random.default_rng(seed).choice(len(src), max_samples, replace=False)
        src, dst = src[keep], dst[keep]
    base = identity(grid)
    lam = smooth * len(src) / grid**3
    eps = 1e-3 * lam + 1e-12
    correction = np.zeros_like(base)
    for _ in range(passes):
        residual = dst - apply(base + correction, src)
        num, den = _splat(grid, src, residual)
        correction += _solve(num, den, lam, eps)
    lut = np.clip(base + correction, 0.0, 1.0)
    _, den = _splat(grid, src, np.zeros_like(src))
    error = dst - apply(lut, src)
    info = {
        "samples": len(src),
        "coverage": float((den > 0.5).mean()),
        "rms": float(np.sqrt((error**2).mean())),
    }
    return lut, info
