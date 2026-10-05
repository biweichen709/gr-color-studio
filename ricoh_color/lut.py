"""3D LUTs as float arrays shaped (N, N, N, 3), indexed [r, g, b] on [0, 1]."""

from pathlib import Path

import numpy as np

UNIT_DOMAIN = {"DOMAIN_MIN": [0.0] * 3, "DOMAIN_MAX": [1.0] * 3, "LUT_3D_INPUT_RANGE": [0.0, 1.0]}


def identity(n):
    axis = np.linspace(0.0, 1.0, n)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def cell(n, rgb):
    """Lower corner index and fractional position of each colour inside the grid."""
    x = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.intp), n - 2)
    return i0, x - i0


def corners(i0, f):
    """Yield ((r, g, b) index arrays, trilinear weight) for the 8 cell corners."""
    for dr in (0, 1):
        wr = f[..., 0] if dr else 1.0 - f[..., 0]
        for dg in (0, 1):
            wg = f[..., 1] if dg else 1.0 - f[..., 1]
            for db in (0, 1):
                wb = f[..., 2] if db else 1.0 - f[..., 2]
                yield (i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db), wr * wg * wb


def apply(lut, rgb):
    """Trilinear lookup of colours (..., 3) through lut."""
    i0, f = cell(lut.shape[0], rgb)
    out = np.zeros(i0.shape, dtype=np.float64)
    for index, weight in corners(i0, f):
        out += lut[index] * weight[..., None]
    return out


def resample(lut, n):
    return apply(lut, identity(n))


def read_cube(path):
    size = None
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, *rest = line.split()
        key = key.upper()
        if key == "TITLE":
            continue
        if key == "LUT_3D_SIZE":
            size = int(rest[0])
        elif key == "LUT_1D_SIZE":
            raise ValueError("1D .cube files are not supported; use a 3D LUT")
        elif key in UNIT_DOMAIN:
            if [float(v) for v in rest] != UNIT_DOMAIN[key]:
                raise ValueError(f"{key} {rest} unsupported; only a 0..1 domain is handled")
        else:
            rows.append([float(v) for v in line.split()[:3]])
    if size is None:
        raise ValueError("missing LUT_3D_SIZE")
    if len(rows) != size**3:
        raise ValueError(f"expected {size**3} rows, found {len(rows)}")
    # .cube rows run with red fastest, so the reshape yields [b][g][r].
    return np.asarray(rows).reshape(size, size, size, 3).transpose(2, 1, 0, 3)


def write_cube(path, lut, title=None):
    n = lut.shape[0]
    lines = [f'TITLE "{title}"'] if title else []
    lines += [f"LUT_3D_SIZE {n}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for r, g, b in lut.transpose(2, 1, 0, 3).reshape(-1, 3):
        lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def hald_identity(level):
    """Identity Hald CLUT image (level**3 square pixels) for a cube of level**2 nodes."""
    n = level * level
    side = level**3
    pixels = identity(n).transpose(2, 1, 0, 3).reshape(side, side, 3)
    return np.rint(pixels * 255).astype(np.uint8)


def hald_to_lut(image):
    """Convert a graded Hald image (float, square, side = level**3) back into a LUT."""
    side = image.shape[0]
    level = round(side ** (1 / 3))
    if image.shape[:2] != (side, side) or level**3 != side:
        raise ValueError("not a Hald CLUT image: expected a square of side level**3")
    n = level * level
    return np.asarray(image, dtype=np.float64).reshape(n, n, n, 3).transpose(2, 1, 0, 3)


def load_image(path, downscale=1):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale), Image.BOX)
        return np.asarray(im, dtype=np.float64) / 255.0


def save_image(path, rgb):
    from PIL import Image

    data = np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(data, "RGB").save(path, quality=95)
