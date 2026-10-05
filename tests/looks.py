import numpy as np

from ricoh_color.lut import identity


def film(n, warm=0.03, sat=1.25, contrast=1.0):
    """A smooth, clipped, film-like test look."""
    x = identity(n)
    y = 0.5 + (x - 0.5) * contrast + 0.08 * np.sin(np.pi * (x - 0.5)) * contrast
    luma = y @ np.array([0.299, 0.587, 0.114])
    y = luma[..., None] + (y - luma[..., None]) * sat + np.array([warm, 0.0, -warm])
    return np.clip(y, 0.0, 1.0)
