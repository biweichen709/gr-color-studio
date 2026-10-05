"""Decode/encode a 3D colour table stored inside an arbitrary binary file.

The camera's real storage format is unknown. A TableSpec describes one
hypothesis (grid, sample type, layout, axis order, byte offset); the scanner
proposes specs and these functions turn bytes into a LUT and back without
changing any byte outside the table or the file length.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DTYPES = {"u8": "u1", "u16le": "<u2", "u16be": ">u2", "u32le": "<u4", "f32le": "<f4"}


@dataclass(frozen=True)
class TableSpec:
    grid: int
    dtype: str = "u16le"
    maxval: float = 4095
    layout: str = "interleaved"  # or "planar": all of channel 0, then 1, then 2
    fastest: str = "r"  # input axis that varies fastest in memory: "r" or "b"
    channels: str = "rgb"  # output channel order inside a node/plane: "rgb" or "bgr"
    pad: int = 0  # extra stored values per node (interleaved only), preserved on encode
    offset: int = 0

    def __post_init__(self):
        if self.grid < 2:
            raise ValueError("grid must be >= 2")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        if self.layout not in ("interleaved", "planar"):
            raise ValueError("layout must be interleaved or planar")
        if self.fastest not in ("r", "b") or self.channels not in ("rgb", "bgr"):
            raise ValueError("fastest must be r/b and channels rgb/bgr")
        if self.pad < 0 or (self.pad and self.layout == "planar"):
            raise ValueError("pad must be >= 0 and is only valid for interleaved tables")
        if self.offset < 0 or self.maxval <= 0:
            raise ValueError("offset must be >= 0 and maxval > 0")

    @property
    def np_dtype(self):
        return np.dtype(DTYPES[self.dtype])

    @property
    def count(self):
        return self.grid**3 * (3 + self.pad)

    @property
    def nbytes(self):
        return self.count * self.np_dtype.itemsize

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def _check_bounds(spec, length):
    if spec.offset + spec.nbytes > length:
        raise ValueError(
            f"table needs bytes {spec.offset}..{spec.offset + spec.nbytes} but file has {length}"
        )


def _to_rgb_axes(stored, spec):
    """(slow, mid, fast, ch) storage view -> [r, g, b, ch] in rgb channel order."""
    a = stored.transpose(2, 1, 0, 3) if spec.fastest == "r" else stored
    return a[..., ::-1] if spec.channels == "bgr" else a


def _from_rgb_axes(lut, spec):
    a = lut[..., ::-1] if spec.channels == "bgr" else lut
    return a.transpose(2, 1, 0, 3) if spec.fastest == "r" else a


def _view(raw, spec):
    n = spec.grid
    if spec.layout == "interleaved":
        return raw.reshape(n, n, n, 3 + spec.pad)[..., :3]
    return raw.reshape(3, n, n, n).transpose(1, 2, 3, 0)


def decode(data, spec):
    _check_bounds(spec, len(data))
    raw = np.frombuffer(data, dtype=spec.np_dtype, count=spec.count, offset=spec.offset)
    return _to_rgb_axes(_view(raw, spec), spec).astype(np.float64) / spec.maxval


def encode(lut, spec, template):
    """Return template with the table region replaced by lut; length is unchanged."""
    n = spec.grid
    lut = np.asarray(lut, dtype=np.float64)
    if lut.shape != (n, n, n, 3):
        raise ValueError(f"LUT shape {lut.shape} does not match grid {n}")
    _check_bounds(spec, len(template))
    dt = spec.np_dtype
    raw = np.frombuffer(template, dtype=dt, count=spec.count, offset=spec.offset).copy()
    values = _from_rgb_axes(lut, spec) * spec.maxval
    if dt.kind == "u":
        values = np.clip(np.rint(values), 0, np.iinfo(dt).max)
    _view(raw, spec)[...] = values.astype(dt)
    out = bytearray(template)
    out[spec.offset : spec.offset + spec.nbytes] = raw.tobytes()
    return bytes(out)
