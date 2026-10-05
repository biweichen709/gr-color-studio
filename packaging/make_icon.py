import sys

import numpy as np
from PIL import Image, ImageDraw


def icon(size=256):
    y, x = np.mgrid[0:size, 0:size] / (size - 1) * 2 - 1
    r = np.hypot(x, y)
    hue = (np.degrees(np.arctan2(y, x)) + 360) % 360 / 360
    k = (np.array([5, 3, 1])[None, None, :] + hue[..., None] * 6) % 6
    rgb = 0.92 - 0.55 * np.clip(np.minimum(k, 4 - k), 0, 1)
    shade = np.clip(1.15 - r * 0.6, 0, 1)[..., None]
    img = np.dstack([rgb * shade * 255, np.where(r <= 0.98, 255, 0)]).astype(np.uint8)
    im = Image.fromarray(img, "RGBA")
    d = ImageDraw.Draw(im)
    c, ring = size / 2, size * 0.30
    d.ellipse((c - ring, c - ring, c + ring, c + ring), fill=(28, 28, 30, 255), outline=(235, 235, 235, 255), width=max(2, size // 40))
    inner = size * 0.12
    d.ellipse((c - inner, c - inner, c + inner, c + inner), fill=(70, 90, 120, 255))
    return im


if __name__ == "__main__":
    icon().save(sys.argv[1], sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
