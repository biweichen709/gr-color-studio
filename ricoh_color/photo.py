from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .lut import cell

RAW_EXTENSIONS = {".dng", ".pef", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CHUNK_PIXELS = 1 << 17

IFD0_TAGS = (0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298)
EXIF_IFD = 0x8769


class Photo:
    def __init__(self, pixels, exif=None, icc=None, orientation=1, source=None):
        self.pixels = pixels
        self.exif = exif
        self.icc = icc
        self.orientation = orientation
        self.source = source

    def preview(self, max_side=1400):
        im = Image.fromarray(self.pixels)
        if self.orientation != 1:
            transpose = ImageOps.exif_transpose
            exif = Image.Exif()
            exif[0x0112] = self.orientation
            im.info["exif"] = exif.tobytes()
            im = transpose(im)
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        return np.asarray(im)


def raw_supported():
    try:
        import rawpy
    except ImportError:
        return False
    return True


def _raw_exif(path):
    try:
        with Image.open(path) as im:
            source = im.getexif()
            out = Image.Exif()
            for tag in IFD0_TAGS:
                if tag in source:
                    out[tag] = source[tag]
            sub = source.get_ifd(EXIF_IFD)
            if sub:
                out.get_ifd(EXIF_IFD).update(
                    {k: v for k, v in sub.items() if not isinstance(v, bytes) or len(v) < 256}
                )
            out[0x0112] = 1
            return out.tobytes()
    except Exception:
        return None


def load(path):
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            pixels = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                auto_bright_thr=0.001,
            )
        return Photo(np.ascontiguousarray(pixels), exif=_raw_exif(path), source=path)
    with Image.open(path) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        orientation = im.getexif().get(0x0112, 1)
        pixels = np.asarray(im.convert("RGB"))
    return Photo(pixels, exif=exif, icc=icc, orientation=orientation, source=path)


def apply_lut(table, pixels):
    n = table.shape[0]
    flat = np.ascontiguousarray(table, dtype=np.float32).reshape(-1, 3) * np.float32(255)
    level, frac = cell(n, np.arange(256) / 255.0)
    level = level.astype(np.int32)
    frac = frac.astype(np.float32)
    sr, sg = n * n, n
    src = pixels.reshape(-1, 3)
    out = np.empty_like(src)
    for start in range(0, len(src), CHUNK_PIXELS):
        px = src[start : start + CHUNK_PIXELS]
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        base = level[r] * sr + level[g] * sg + level[b]
        fr, fg, fb = frac[r][:, None], frac[g][:, None], frac[b][:, None]

        def lerp_b(i):
            lo = np.take(flat, i, axis=0)
            hi = np.take(flat, i + 1, axis=0)
            hi -= lo
            hi *= fb
            hi += lo
            return hi

        def lerp_g(i):
            lo = lerp_b(i)
            hi = lerp_b(i + sg)
            hi -= lo
            hi *= fg
            hi += lo
            return hi

        lo = lerp_g(base)
        acc = lerp_g(base + sr)
        acc -= lo
        acc *= fr
        acc += lo
        out[start : start + CHUNK_PIXELS] = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
    return out.reshape(pixels.shape)


def save_jpeg(path, pixels, photo=None, quality=95):
    kwargs = {"quality": quality, "subsampling": 0, "optimize": True}
    if photo is not None:
        if photo.exif:
            kwargs["exif"] = photo.exif
        if photo.icc:
            kwargs["icc_profile"] = photo.icc
    Image.fromarray(pixels, "RGB").save(path, "JPEG", **kwargs)


def is_supported(path):
    suffix = Path(path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS or (suffix in RAW_EXTENSIONS and raw_supported())
