import re
from functools import lru_cache
from itertools import permutations

import numpy as np

from .codec import DTYPES, TableSpec

KEYWORDS = (
    "positive", "negative", "bleach", "retro", "crossproc", "cross proc", "cross_proc",
    "monotone", "hicontrast", "hi-contrast", "hi_contrast", "hdrtone", "hdr tone", "cinema",
    "imagecontrol", "image control", "image_control", "imgctrl", "effect", "lut", "gamma",
    "tonecurve", "tone curve", "tone_curve", "colormatrix", "color matrix", "ccm",
    "saturation", "film",
)
DRIVE_PATH = re.compile(r"\b[A-Z]:\\")

GRIDS = (9, 16, 17, 32, 33, 65)
SCAN_DTYPES = ("u8", "u16le", "u16be")
LAYOUTS = (("interleaved", 0), ("interleaved", 1), ("planar", 0))
BLOCK = 768
CHUNK_BLOCKS = 4096
MIN_STRUCTURE = 0.5
MIN_WRAP_SHARE = 0.5
OUTLIER_BEND = 0.4
MAXVALS = (255, 1023, 4095, 16383, 65535)


def find_strings(data, min_len=4):
    found = []
    for pattern, encoding in (
        (rb"[\x20-\x7e]{%d,}" % min_len, "ascii"),
        (rb"(?:[\x20-\x7e]\x00){%d,}" % min_len, "utf-16-le"),
    ):
        for m in re.finditer(pattern, data):
            found.append((m.start(), encoding, m.group().decode(encoding)))
    return sorted(found)


def interesting_strings(data, extra=()):
    words = tuple(w.lower() for w in KEYWORDS + tuple(extra))
    for offset, encoding, text in find_strings(data):
        low = text.lower()
        hits = [w for w in words if w in low]
        path = bool(DRIVE_PATH.search(text))
        if hits or path:
            yield {"offset": offset, "encoding": encoding, "text": text, "keywords": hits, "path": path}


def _geometry(n, layout, pad):
    if layout == "interleaved":
        stride = 3 + pad
        return stride, n * stride, n**3 * stride
    return 1, n, 3 * n**3


def _max_rough(n):
    return 8.0 / (n - 1) ** 2


def _smooth_blocks(values, stride):
    nblocks = len(values) // BLOCK
    flags = np.zeros(nblocks, dtype=bool)
    for first in range(0, nblocks, CHUNK_BLOCKS):
        last = min(nblocks, first + CHUNK_BLOCKS)
        v = values[first * BLOCK : last * BLOCK].astype(np.float64).reshape(-1, BLOCK)
        span = v.max(1) - v.min(1)
        bend = np.median(np.abs(v[:, 2 * stride :] - 2 * v[:, stride:-stride] + v[:, : -2 * stride]), axis=1)
        distinct = (np.diff(np.sort(v, axis=1), axis=1) > 0).sum(1) + 1
        flags[first:last] = (span > 0) & (bend <= 0.1 * span) & (distinct >= 5)
    return flags


def _runs(flags):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _storage(values, start, n, layout, pad, planes):
    if layout == "interleaved":
        width = 3 + pad
        window = values[start : start + planes * n * n * width]
        return window.reshape(planes, n, n, width)[..., :3].astype(np.float64)
    size = n**3
    parts = [values[start + c * size : start + c * size + planes * n * n] for c in range(3)]
    return np.stack(parts, axis=-1).reshape(planes, n, n, 3).astype(np.float64)


@lru_cache(maxsize=32)
def _centred_coords(shape):
    coords = np.stack(
        np.meshgrid(*(np.arange(m, dtype=np.float64) for m in shape), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    coords -= coords.mean(0)
    return coords, np.sqrt((coords**2).mean(0))


def _correlations(s):
    y, sy = _centred_coords(s.shape[:3])
    x = s.reshape(-1, 3)
    x = x - x.mean(0)
    denom = np.outer(np.sqrt((x**2).mean(0)), sy)
    cov = x.T @ y / len(x)
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)


def _score(s, structure=True):
    lo, hi = np.percentile(s, [1, 99])
    span = hi - lo
    if span <= 0:
        return np.inf, np.inf, 0.0
    bends = [np.abs(np.diff(s, 2, axis=a)) for a in range(3) if s.shape[a] >= 3]
    outliers = sum(int((b > OUTLIER_BEND * span).sum()) for b in bends)
    rough = sum(b.mean() for b in bends) / span
    return outliers, rough, _assignment(_correlations(s))[1] if structure else 1.0


def _assignment(corr):
    best = max(permutations(range(3)), key=lambda p: min(corr[p[a], a] for a in range(3)))
    return best, float(min(corr[best[a], a] for a in range(3)))


def _wrap_phase(drops, period, quantile):
    rows = len(drops) // period
    if rows < 2:
        return None
    profile = np.percentile(drops[: rows * period].reshape(rows, period), quantile, axis=0)
    total = profile.sum()
    if total == 0 or profile.min() / total < MIN_WRAP_SHARE:
        return None
    return int(np.argmin(profile))


def _row_phases(values, lo, hi, n, layout, pad):
    stride, row, _ = _geometry(n, layout, pad)
    v = values[lo:hi].astype(np.float64)
    quantile = 50 if layout == "interleaved" else 20
    p = _wrap_phase(np.minimum(v[stride:] - v[:-stride], 0), row, quantile)
    if p is None:
        return None
    lanes = range(3) if layout == "interleaved" else range(1)
    return sorted({(lo + p + stride - c) % row for c in lanes})


def _plane_phase(values, first, r0, r1, n, layout, row):
    lanes = 3 if layout == "interleaved" else 1
    heads = first + np.arange(r0, r1)[:, None] * row + np.arange(lanes)[None, :]
    drops = np.minimum(np.diff(values[heads].astype(np.float64), axis=0), 0).min(axis=1)
    quantile = 50 if layout == "interleaved" else 20
    p = _wrap_phase(drops, n, quantile)
    return None if p is None else (r0 + p + 1) % n


def _clean_rows(values, first, count, n, stride, span):
    w = values[first : first + count * n * stride].astype(np.float64)
    bend = np.abs(np.diff(w.reshape(count, n, stride)[..., :3], 2, axis=1))
    return bend.max(axis=(1, 2)) <= OUTLIER_BEND * span


def _candidate(values, start, score, n, layout, pad, dtype):
    outliers, rough, structure = score
    full = _storage(values, start, n, layout, pad, n)
    fast_channel = int(_assignment(_correlations(full))[0][2])
    order = {0: (("r", "rgb"), ("b", "bgr")), 2: (("b", "rgb"), ("r", "bgr"))}
    (fastest, channels), (alt_fastest, alt_channels) = order.get(fast_channel, order[0])
    top = float(full.max())
    maxval = next((m for m in MAXVALS if m >= top), MAXVALS[-1])
    spec = TableSpec(
        grid=n, dtype=dtype, maxval=maxval, layout=layout, fastest=fastest,
        channels=channels, pad=pad, offset=int(start) * values.dtype.itemsize,
    )
    return {
        "kind": "lut3d",
        "offset": spec.offset,
        "nbytes": spec.nbytes,
        "outliers": int(outliers),
        "rough": round(float(rough), 5),
        "structure": round(structure, 3),
        "value_range": [float(full.min()), top],
        "spec": spec.to_dict(),
        "alternative": {"fastest": alt_fastest, "channels": alt_channels},
        "note": "" if fast_channel in order else "unusual channel order; inspect manually",
    }


def _search_region(values, lo, hi, n, layout, pad, dtype):
    stride, row, total = _geometry(n, layout, pad)
    a, b = max(0, lo - BLOCK), min(len(values), hi + BLOCK)
    if b - a < total:
        return []
    phases = _row_phases(values, lo, hi, n, layout, pad)
    if phases is None:
        return []
    lo_q, hi_q = np.percentile(values[a:b], [1, 99])
    span = hi_q - lo_q
    if span <= 0:
        return []
    rows_per_table = total // row
    limit = max(2, total // 1000)
    max_rough = _max_rough(n)

    def acceptable(score):
        return score[0] <= limit and score[1] <= max_rough and score[2] >= MIN_STRUCTURE

    hits = []
    for phase in phases:
        first = a + (phase - a) % row
        count = (b - first) // row
        for r0, r1 in _runs(_clean_rows(values, first, count, n, stride, span)):
            if r1 - r0 < rows_per_table:
                continue
            plane = _plane_phase(values, first, r0, r1, n, layout, row)
            if plane is None:
                continue
            k = r0 + (plane - r0) % n
            while k + rows_per_table <= r1:
                start = first + k * row
                preview = _score(_storage(values, start, n, layout, pad, min(n, 4)), structure=False)
                if acceptable(preview):
                    score = _score(_storage(values, start, n, layout, pad, n))
                    if acceptable(score):
                        hits.append(_candidate(values, start, score, n, layout, pad, dtype))
                        k += rows_per_table
                        continue
                k += n
    return hits


def _dedupe(hits):
    kept = []
    for hit in sorted(hits, key=lambda h: (h["outliers"], h["rough"], -h["structure"])):
        lo, hi = hit["offset"], hit["offset"] + hit["nbytes"]
        if all(hi <= k["offset"] or lo >= k["offset"] + k["nbytes"] for k in kept):
            kept.append(hit)
    return sorted(kept, key=lambda h: h["offset"])


def find_luts(data, grids=GRIDS, dtypes=SCAN_DTYPES):
    hits = []
    for dtype in dtypes:
        dt = np.dtype(DTYPES[dtype])
        values = np.frombuffer(data, dtype=dt, count=len(data) // dt.itemsize)
        for layout, pad in LAYOUTS:
            stride = _geometry(2, layout, pad)[0]
            for b0, b1 in _runs(_smooth_blocks(values, stride)):
                for n in grids:
                    hits += _search_region(values, b0 * BLOCK, b1 * BLOCK, n, layout, pad, dtype)
    return _dedupe(hits)


def find_curves(data, dtypes=SCAN_DTYPES, min_len=256):
    hits = []
    for dtype in dtypes:
        dt = np.dtype(DTYPES[dtype])
        v = np.frombuffer(data, dtype=dt, count=len(data) // dt.itemsize)
        if len(v) < min_len:
            continue
        rising = np.concatenate([[0], (v[1:] >= v[:-1]).astype(np.int8), [0]])
        edges = np.flatnonzero(np.diff(rising))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a + 1 < min_len:
                continue
            seg = v[a : b + 1].astype(np.float64)
            steps = np.diff(seg)
            distinct = int(np.count_nonzero(steps)) + 1
            if distinct < 32 or seg[-1] - seg[0] < (64 if dtype == "u8" else 256):
                continue
            hits.append({
                "kind": "curve",
                "offset": int(a) * dt.itemsize,
                "dtype": dtype,
                "length": int(b - a + 1),
                "first": float(seg[0]),
                "last": float(seg[-1]),
                "distinct": distinct,
                "linear": bool(steps.max() - steps.min() <= 1),
            })
    return hits
