"""Node-wise retargeting of a camera colour table.

The camera table keeps its grid: node (i, j, k) stays node (i, j, k). Only the
value stored at each node changes, from base[i, j, k] to look(base[i, j, k]).
Because the input axes are untouched, the table's input colour space does not
need to be known; only the space of its *output* values does (``space``).
"""

import numpy as np

from . import color
from .lut import apply

# Converters (table output -> display sRGB, display sRGB -> table output).
SPACES = {
    "rgb": (lambda v: v, lambda v: v),
    "linear": (color.linear_to_srgb, color.srgb_to_linear),
    "ycbcr": (color.ycbcr_to_rgb, color.rgb_to_ycbcr),
}


def remap(base, look, space="rgb", strength=1.0):
    """Return base with look applied to every node value, blended by strength."""
    to_display, from_display = SPACES[space]
    shown = np.clip(to_display(base), 0.0, 1.0)
    target = shown + strength * (apply(look, shown) - shown)
    # Apply as a delta so nodes the look leaves alone keep their exact stored value.
    out = base + (from_display(target) - from_display(shown))
    return np.clip(out, min(0.0, base.min()), max(1.0, base.max()))
