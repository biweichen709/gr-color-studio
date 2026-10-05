import numpy as np

from . import color
from .lut import apply

SPACES = {
    "rgb": (lambda v: v, lambda v: v),
    "linear": (color.linear_to_srgb, color.srgb_to_linear),
    "ycbcr": (color.ycbcr_to_rgb, color.rgb_to_ycbcr),
}


def remap(base, look, space="rgb", strength=1.0):
    to_display, from_display = SPACES[space]
    shown = np.clip(to_display(base), 0.0, 1.0)
    target = shown + strength * (apply(look, shown) - shown)
    out = base + (from_display(target) - from_display(shown))
    return np.clip(out, min(0.0, base.min()), max(1.0, base.max()))
