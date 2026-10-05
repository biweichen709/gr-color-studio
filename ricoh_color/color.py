"""Colour conversions (sRGB / D65) and CIEDE2000."""

import numpy as np

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
D65 = np.array([0.95047, 1.0, 1.08883])

# JPEG (BT.601 full-range) YCbCr, with Cb/Cr centred on 0.5.
RGB_TO_YCC = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)
YCC_TO_RGB = np.linalg.inv(RGB_TO_YCC)
YCC_OFFSET = np.array([0.0, 0.5, 0.5])


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055)


def rgb_to_ycbcr(rgb):
    return rgb @ RGB_TO_YCC.T + YCC_OFFSET


def ycbcr_to_rgb(ycc):
    return (ycc - YCC_OFFSET) @ YCC_TO_RGB.T


def srgb_to_lab(rgb):
    xyz = srgb_to_linear(rgb) @ SRGB_TO_XYZ.T / D65
    eps = (6 / 29) ** 3
    f = np.where(xyz > eps, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def delta_e2000(lab1, lab2):
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma_zero = c1p * c2p == 0

    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(chroma_zero, 0.0, dh)
    d_l = L2 - L1
    d_c = c2p - c1p
    d_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    l_bar = (L1 + L2) / 2
    cp_bar = (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(
        np.abs(h1p - h2p) > 180,
        np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2),
        h_sum / 2,
    )
    h_bar = np.where(chroma_zero, h_sum, h_bar)

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * cp_bar
    s_h = 1 + 0.015 * cp_bar * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c
    return np.sqrt(
        (d_l / s_l) ** 2 + (d_c / s_c) ** 2 + (d_h / s_h) ** 2 + r_t * (d_c / s_c) * (d_h / s_h)
    )


def delta_e_stats(rgb_a, rgb_b):
    de = delta_e2000(srgb_to_lab(rgb_a), srgb_to_lab(rgb_b)).ravel()
    return {
        "mean": float(de.mean()),
        "p95": float(np.percentile(de, 95)),
        "max": float(de.max()),
    }
