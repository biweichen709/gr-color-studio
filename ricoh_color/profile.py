"""Detect the parametric forms that encode in-camera "image control / filter" color.

A 3D LUT is only one way to store a look. Many cameras (GR III included, on the
evidence) render filters with a 3x3 colour matrix (CCM) plus a 1D tone curve and
a few scalars per mode, compiled as small tables. Those are invisible to the 3D
LUT scanner (9 numbers, short curves), so this module adds matrix / curve-bank /
mode-block detectors, anchored to the filter mode-name strings, and reports WHERE
each hit lives so firmware-baked (not replaceable) can be told apart from an
A:-resident file (the one real lever for SD-card replacement).
"""

import re

import numpy as np

from .color import RGB_TO_YCC, SRGB_TO_XYZ
from .scan import find_strings

# Image-control / filter mode names, in canonical-ish order. ASCII is how Ricoh
# labels these internally in the GR IV family; UTF-16LE is covered by find_strings.
MODE_LEXICON = [
    "Standard", "Vivid", "Natural", "Monotone", "Soft Monotone", "Hard Monotone",
    "Hi-Contrast B&W", "Hi Contrast", "Positive Film", "Bleach Bypass", "Bleach",
    "Retro", "HDR Tone", "Cross Process", "Cross", "Custom", "Neutral", "Portrait",
    "Landscape", "Bright", "Muted",
]
CALIB_NEAR = ("DarkCurr", "Awb_Gain", "Sr_", "BlkCtl", "DF_", "PxDf", "MfgPrc", "Sen_")
ANCHOR_WINDOW = 1 << 16  # bytes around a mode-name cluster to search
FIXED_SCALES = (256, 512, 1024, 2048, 4096, 8192, 16384)
PIPELINE_MATRICES = (RGB_TO_YCC, SRGB_TO_XYZ, np.linalg.inv(SRGB_TO_XYZ))


# ---------------------------------------------------------------- anchors


def mode_anchors(data):
    """Sorted [(offset, name)] of filter mode-name strings (ASCII + UTF-16LE)."""
    low = {m.lower() for m in MODE_LEXICON}
    hits = []
    for offset, _, text in find_strings(data, min_len=4):
        t = text.strip().lower()
        for name in MODE_LEXICON:
            if name.lower() == t or name.lower() in t:
                hits.append((offset, name))
                break
    return sorted(set(hits))


def clusters(anchors, gap=4096, min_names=2):
    """Group nearby anchors into (lo, hi, names); keep clusters of >= min_names."""
    out = []
    run = []
    for offset, name in anchors:
        if run and offset - run[-1][0] > gap:
            if len({n for _, n in run}) >= min_names:
                out.append((run[0][0], run[-1][0], [n for _, n in run]))
            run = []
        run.append((offset, name))
    if run and len({n for _, n in run}) >= min_names:
        out.append((run[0][0], run[-1][0], [n for _, n in run]))
    return out


def _near_calib(data, offset, radius=4096):
    window = data[max(0, offset - radius): offset + radius]
    return any(w.encode() in window for w in CALIB_NEAR)


# ---------------------------------------------------------------- 3x3 colour matrices


def _valid_ccm(m):
    if not np.all(np.isfinite(m)):
        return False
    if np.abs(m).max() > 2.5 or np.abs(np.diag(m)).min() < 0.3:
        return False
    if not np.all(np.abs(m.sum(axis=1) - 1.0) < 0.08):  # white-preserving rows
        return False
    det = abs(float(np.linalg.det(m)))
    if not 0.1 <= det <= 10:
        return False
    for p in PIPELINE_MATRICES:  # exclude hardware colour-space conversions
        if m.shape == p.shape and np.abs(m - p).max() < 0.02:
            return False
    return True


def _candidate_mask(w9):
    """Cheap white-preserving/bounded prefilter over rows of 9 values (N, 9)."""
    finite = np.isfinite(w9).all(axis=1)
    bounded = np.abs(w9).max(axis=1) <= 2.5
    rows = w9.reshape(-1, 3, 3)
    white = np.abs(rows.sum(axis=-1) - 1.0).max(axis=-1) < 0.06
    diag = np.minimum.reduce([np.abs(rows[:, 0, 0]), np.abs(rows[:, 1, 1]), np.abs(rows[:, 2, 2])]) >= 0.3
    return finite & bounded & white & diag


def _confirm_mask(m):
    """Expensive det + pipeline-exclusion over candidate matrices (N, 3, 3)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        det = np.abs(np.linalg.det(m))
    ok = (det >= 0.1) & (det <= 10)
    for p in PIPELINE_MATRICES:
        ok &= np.abs(m - p).max(axis=(-2, -1)) >= 0.02
    return ok


def detect_ccm(data, lo, hi, anchors_near):
    """Find banks of white-preserving 3x3 matrices in data[lo:hi] (vectorised)."""
    from numpy.lib.stride_tricks import sliding_window_view

    hits = []
    attempts = [(np.dtype("<f4"), 1.0)] + [(np.dtype("<i2"), s) for s in FIXED_SCALES] + [(np.dtype("<i4"), s) for s in FIXED_SCALES]
    for dtype, scale in attempts:
        size = dtype.itemsize
        start = lo - (lo % size)
        count = (hi - start) // size
        if count < 9:
            continue
        with np.errstate(invalid="ignore"):
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=start).astype(np.float64)
        if dtype.kind != "f":
            arr /= scale
        w9 = sliding_window_view(arr, 9)
        cand = np.flatnonzero(_candidate_mask(w9))
        if not len(cand):
            continue
        confirmed = cand[_confirm_mask(w9[cand].reshape(-1, 3, 3))]
        valid = np.zeros(len(w9), dtype=bool)
        valid[confirmed] = True
        used = -1
        for s0 in confirmed:
            if s0 <= used:
                continue
            bank = 1
            while s0 + bank * 9 + 8 < len(valid) and valid[s0 + bank * 9]:
                bank += 1
            byte_off = start + int(s0) * size
            near = min((abs(byte_off - a) for a in anchors_near), default=None)
            anchored = near is not None and near <= ANCHOR_WINDOW
            if bank >= 4 or (anchored and bank >= 2):
                hits.append({
                    "kind": "ccm", "offset": byte_off, "dtype": dtype.name, "scale": scale,
                    "count": bank, "anchor_distance": near,
                    "matrix0": w9[s0].reshape(3, 3).round(4).tolist(),
                })
                used = s0 + bank * 9 - 1
    return _dedupe(hits)


# ---------------------------------------------------------------- tone-curve banks


CURVE_LENS = (33, 65, 129, 256, 257, 512, 1024, 1025)
CURVE_DTYPES = (("u8", "<u1", 255), ("u16le", "<u2", 65535), ("u16be", ">u2", 65535))


def _curve_runs(values, maxv):
    """Maximal non-decreasing runs that look like tone curves: (start, length)."""
    signed = values.astype(np.int64)  # unsigned diff would wrap and hide the drops
    rising = np.concatenate([[0], (np.diff(signed) >= 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(rising))
    runs = []
    for a, b in zip(edges[::2], edges[1::2]):
        length = b - a + 1
        if length < CURVE_LENS[0]:
            continue
        seg = values[a:b + 1].astype(np.float64)
        distinct = int(np.count_nonzero(np.diff(seg))) + 1
        if (distinct >= 16 and seg[-1] - seg[0] >= 0.4 * maxv
                and seg[0] <= 0.15 * maxv and seg[-1] >= 0.75 * maxv):
            runs.append((int(a), int(length)))
    return runs


def detect_curve_banks(data, lo, hi, anchors_near):
    """Find banks of equal-length tone curves (luma / per-channel / mono triples)."""
    hits = []
    lengths = set(CURVE_LENS)
    for name, dt, maxv in CURVE_DTYPES:
        dtype = np.dtype(dt)
        size = dtype.itemsize
        values = np.frombuffer(data, dtype=dtype, count=len(data) // size)
        lo_i, hi_i = lo // size, hi // size
        runs = [r for r in _curve_runs(values, maxv) if lo_i <= r[0] < hi_i and r[1] in lengths]
        used = -1
        for i, length in runs:
            if i <= used:
                continue
            bank = 1
            while any(j == i + bank * length and length == le for j, le in runs):
                bank += 1
            byte_off = i * size
            near = min((abs(byte_off - a) for a in anchors_near), default=None)
            if bank >= 3 or (near is not None and near <= ANCHOR_WINDOW):
                mono_triple = bank >= 3 and all(
                    np.array_equal(values[i:i + length], values[i + k * length:i + (k + 1) * length])
                    for k in (1, 2))
                hits.append({
                    "kind": "curve_bank", "offset": byte_off, "dtype": name, "length": length,
                    "count": bank, "anchor_distance": near,
                    "near_calibration": _near_calib(data, byte_off),
                    "role": "mono_triple" if mono_triple else ("rgb" if bank % 3 == 0 else "luma"),
                })
            used = i + bank * length - 1
    return _dedupe(hits)


# ---------------------------------------------------------------- mode-indexed blocks


def detect_mode_blocks(data, lo, hi, mode_count):
    """Find a repeated fixed-stride struct array of exactly mode_count entries."""
    if mode_count < 3:
        return []
    region = np.frombuffer(data, dtype=np.uint8, count=hi - lo, offset=lo).astype(np.float64)
    hits = []
    for stride in range(8, 513, 4):
        total = stride * mode_count
        if total > len(region):
            break
        block = region[:total].reshape(mode_count, stride)
        col_std = block.std(axis=0)
        constant = (col_std < 0.5).mean()  # type tags / reserved
        varying = ((col_std > 1.0) & (col_std < 48)).mean()  # small-signed tunables
        if 0.25 <= constant <= 0.9 and varying >= 0.1:
            hits.append({"kind": "mode_block", "offset": lo, "stride": stride,
                         "count": mode_count, "constant_frac": round(float(constant), 2),
                         "varying_frac": round(float(varying), 2)})
    return hits[:3]


# ---------------------------------------------------------------- orchestration


def _dedupe(hits):
    seen, out = set(), []
    for h in sorted(hits, key=lambda h: (h["offset"], -h.get("count", 1))):
        if h["offset"] not in seen:
            seen.add(h["offset"])
            out.append(h)
    return out


def profile_scan(data, whole_file=False):
    """Scan for CCM / curve / mode-block structures, anchored to mode names.

    whole_file=True also scans the entire buffer (use for small backup files);
    otherwise only windows around mode-name clusters are searched (for firmware).
    """
    anchors = mode_anchors(data)
    anchor_offsets = [o for o, _ in anchors]
    cls = clusters(anchors)
    windows = [(max(0, lo - ANCHOR_WINDOW), min(len(data), hi + ANCHOR_WINDOW), names) for lo, hi, names in cls]
    if whole_file or not windows:
        windows.append((0, len(data), [n for _, n in anchors]))
    ccm, curves, blocks = [], [], []
    for lo, hi, names in windows:
        ccm += detect_ccm(data, lo, hi, anchor_offsets)
        curves += detect_curve_banks(data, lo, hi, anchor_offsets)
        blocks += detect_mode_blocks(data, lo, hi, len(set(names)))
    return {
        "anchors": [{"offset": o, "name": n} for o, n in anchors],
        "clusters": [{"lo": lo, "hi": hi, "modes": sorted(set(names))} for lo, hi, names in cls],
        "ccm": _dedupe(ccm),
        "curves": _dedupe(curves),
        "mode_blocks": blocks,
    }


def diff_archives(archive_a, archive_b):
    """Byte-compare two backup archives; return which captured files differ.

    The decisive test: back up the suspect files with the camera in two image
    control modes (e.g. Standard vs Vivid). A file that differs between modes is
    where that look is stored. If nothing on A: differs, the look is firmware-baked.
    """
    import hashlib
    import json
    from pathlib import Path

    def index(root):
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        out = {}
        for e in manifest["entries"]:
            if e["status"] == "OK":
                p = Path(root) / "backup" / e["file"]
                out[e["target"]] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)
        return out

    a, b = index(archive_a), index(archive_b)
    changed, same, only = [], [], []
    for target in sorted(set(a) | set(b)):
        if target in a and target in b:
            (changed if a[target] != b[target] else same).append(target)
        else:
            only.append(target)
    return {"changed": changed, "unchanged_count": len(same), "only_in_one": only}


# ---------------------------------------------------------------- bake-down retarget

from .lut import apply as _lut_apply, identity as _identity  # noqa: E402

_DT = {"float32": np.dtype("<f4"), "int16": np.dtype("<i2"), "int32": np.dtype("<i4")}


def fit_ccm(look, samples=4096, seed=0):
    """Best white-preserving 3x3 matrix approximating a look (display sRGB).

    Returns (matrix, delta_e_stats). A 3x3 cannot carry hue-crossover film looks
    exactly; the reported CIEDE2000 says how lossy the approximation is.
    """
    from .color import delta_e_stats

    rng = np.random.default_rng(seed)
    rgb = rng.random((samples, 3))
    target = _lut_apply(look, rgb)
    rows = []
    for c in range(3):  # min ||rgb @ r - target[:,c]||^2  s.t. sum(r) = 1
        a = rgb.T @ rgb
        b = rgb.T @ target[:, c]
        kkt = np.zeros((4, 4))
        kkt[:3, :3] = a
        kkt[:3, 3] = kkt[3, :3] = 1.0
        rhs = np.append(b, 1.0)
        rows.append(np.linalg.solve(kkt, rhs)[:3])
    m = np.array(rows)
    approx = np.clip(rgb @ m.T, 0, 1)
    return m, delta_e_stats(approx, np.clip(target, 0, 1))


def fit_curve(look, length, channel=None):
    """1D transfer curve of a look: luma response, or one channel's response."""
    x = np.linspace(0, 1, length)
    ramp = np.repeat(x[:, None], 3, axis=1)
    out = _lut_apply(look, ramp)
    if channel is None:
        return np.clip(out @ np.array([0.2126, 0.7152, 0.0722]), 0, 1)
    return np.clip(out[:, channel], 0, 1)


def encode_matrix(data, spec, matrix):
    """Splice one 3x3 matrix into data at spec (offset/dtype/scale/index); length kept."""
    dt = _DT[spec["dtype"]]
    base = spec["offset"] + spec.get("index", 0) * 9 * dt.itemsize
    vals = np.asarray(matrix, dtype=np.float64).reshape(9)
    if dt.kind == "f":
        enc = vals.astype(dt)
    else:
        enc = np.clip(np.rint(vals * spec["scale"]), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    out = bytearray(data)
    out[base:base + 9 * dt.itemsize] = enc.tobytes()
    return bytes(out)


def encode_curve(data, spec, curve):
    """Splice one tone curve (length/dtype) into data at spec; length kept."""
    dt = np.dtype({"u8": "<u1", "u16le": "<u2", "u16be": ">u2"}[spec["dtype"]])
    maxv = 255 if dt.itemsize == 1 else 65535
    base = spec["offset"] + spec.get("index", 0) * spec["length"] * dt.itemsize
    enc = np.clip(np.rint(np.asarray(curve, dtype=np.float64) * maxv), 0, maxv).astype(dt)
    if len(enc) != spec["length"]:
        raise ValueError("curve length does not match the spec")
    out = bytearray(data)
    out[base:base + spec["length"] * dt.itemsize] = enc.tobytes()
    return bytes(out)
