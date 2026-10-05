"""Parametric colour looks, baked into 3D LUTs.

Presets are hand-tuned approximations of how these styles are commonly
described (natural skin and soft roll-off for HNCS; the published character
of each Fujifilm film simulation). They contain no Hasselblad, Fujifilm or
Ricoh data and are not affiliated with those companies.

Pipeline on display (sRGB) values: white balance in linear light, optional
B&W channel mix, monotone tone curves, then hue/saturation/toning edits in
OkLCh, and finally a gamut map that keeps lightness and hue.
"""

from dataclasses import dataclass, replace

import numpy as np

from .color import linear_to_srgb, srgb_to_linear
from .lut import apply, identity

# Oklab hue (degrees) of each adjustable band; sRGB red is ~29, skin ~55.
BANDS = {
    "red": 25, "orange": 55, "yellow": 100, "green": 140,
    "cyan": 195, "blue": 255, "purple": 295, "magenta": 335,
}
BAND_SIGMA = 24.0
LUMA = np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------- building blocks


def monotone_curve(points):
    """Monotone cubic (PCHIP) through (x, y) control points; returns a vectorised f."""
    xs, ys = (np.asarray(v, dtype=np.float64) for v in zip(*sorted(points)))
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty_like(xs)
    m[0], m[-1] = delta[0], delta[-1]
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def f(v):
        v = np.clip(v, xs[0], xs[-1])
        k = np.clip(np.searchsorted(xs, v, side="right") - 1, 0, len(xs) - 2)
        t = (v - xs[k]) / h[k]
        t2, t3 = t * t, t * t * t
        return (
            (2 * t3 - 3 * t2 + 1) * ys[k]
            + (t3 - 2 * t2 + t) * h[k] * m[k]
            + (-2 * t3 + 3 * t2) * ys[k + 1]
            + (t3 - t2) * h[k] * m[k + 1]
        )

    return f


def linear_to_oklab(rgb):
    lms = rgb @ np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]).T
    return np.cbrt(lms) @ np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]).T


def oklab_to_linear(lab):
    lms = (lab @ np.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]).T) ** 3
    return lms @ np.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]).T


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _gamut_map(lin, lightness):
    """Pull out-of-range colours toward the grey of equal Oklab lightness."""
    grey = np.clip(lightness, 0.0, 1.0)[..., None] ** 3
    d = lin - grey
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(d > 1e-12, (1.0 - grey) / d, np.inf)
        down = np.where(d < -1e-12, -grey / d, np.inf)
    t = np.minimum(1.0, np.minimum(up, down).min(axis=-1, keepdims=True))
    return np.clip(grey + t * d, 0.0, 1.0)


# ---------------------------------------------------------------- looks


@dataclass(frozen=True)
class Look:
    key: str
    name: str
    group: str
    note: str = ""
    curve: tuple = ((0.0, 0.0), (1.0, 1.0))  # master tone curve on display values
    curve_luma: float = 0.0  # 0 = per channel (film-like), 1 = lightness only
    rgb_curves: tuple = (None, None, None)  # optional extra curve per channel
    temp: float = 0.0  # + warmer
    tint: float = 0.0  # + magenta
    saturation: float = 1.0
    vibrance: float = 0.0
    shadow_sat: float = 1.0
    highlight_sat: float = 1.0
    bands: tuple = ()  # (band, hue shift deg, saturation x, lightness +)
    shadow_tint: tuple = (0.0, 0.0)  # (Oklab hue deg, amount)
    highlight_tint: tuple = (0.0, 0.0)
    mono: tuple = None  # B&W channel weights (linear light)


@dataclass(frozen=True)
class Adjust:
    """User fine-tuning on top of a preset; all zero/one means 'as designed'."""

    strength: float = 1.0
    temp: float = 0.0
    tint: float = 0.0
    contrast: float = 0.0
    saturation: float = 1.0
    fade: float = 0.0

    def is_neutral(self):
        return self == Adjust()


def _white_balance(display, temp, tint):
    if not temp and not tint:
        return display
    gains = np.exp(np.array([temp * 0.35 + tint * 0.12, -tint * 0.24, -temp * 0.35 + tint * 0.12]))
    gains /= gains @ LUMA
    return linear_to_srgb(srgb_to_linear(display) * gains)


def _tone(display, look, adjust):
    curves = [monotone_curve(look.curve)]
    if adjust.contrast:
        k = 0.07 * adjust.contrast
        curves.append(monotone_curve(((0, 0), (0.25, 0.25 - k), (0.5, 0.5), (0.75, 0.75 + k), (1, 1))))
    if adjust.fade:
        curves.append(lambda v, f=adjust.fade * 0.12: f + (1 - f * 0.5) * v)

    def master(v):
        for c in curves:
            v = c(v)
        return v

    out = master(display)
    if look.curve_luma:
        y = display @ LUMA
        out = (1 - look.curve_luma) * out + look.curve_luma * (display + (master(y) - y)[..., None])
    for c, points in enumerate(look.rgb_curves):
        if points:
            out[..., c] = monotone_curve(points)(out[..., c])
    return out


def _band_weights(hue):
    centres = np.array(list(BANDS.values()), dtype=np.float64)
    d = (hue[..., None] - centres + 180.0) % 360.0 - 180.0
    w = np.exp(-0.5 * (d / BAND_SIGMA) ** 2)
    return w / w.sum(axis=-1, keepdims=True)


def _colour(lab, look, adjust):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0

    if look.bands:
        names = list(BANDS)
        table = np.zeros((len(names), 3))
        table[:, 1] = 1.0
        for band, dh, sat, light in look.bands:
            table[names.index(band)] = (dh, sat, light)
        w = _band_weights(h) @ table
        chroma_w = _smoothstep(0.0, 0.06, C)
        h = h + w[..., 0] * chroma_w
        C = C * (1 + (w[..., 1] - 1) * chroma_w)
        L = L + w[..., 2] * chroma_w * np.minimum(1.0, C / 0.1)

    C = C * look.saturation * adjust.saturation
    if look.vibrance:
        C = C * (1 + look.vibrance * (1 - np.minimum(C / 0.25, 1.0)))
    if look.shadow_sat != 1.0 or look.highlight_sat != 1.0:
        lo, hi = _smoothstep(0.0, 0.55, L), _smoothstep(0.55, 1.0, L)
        C = C * (look.shadow_sat + (1 - look.shadow_sat) * lo) * (1 + (look.highlight_sat - 1) * hi)

    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    for (hue, amount), weight in (
        (look.shadow_tint, 1 - _smoothstep(0.0, 0.65, L)),
        (look.highlight_tint, _smoothstep(0.35, 1.0, L)),
    ):
        if amount:
            a = a + amount * weight * np.cos(np.radians(hue))
            b = b + amount * weight * np.sin(np.radians(hue))
    return np.stack([L, a, b], axis=-1)


def render(look, display, adjust=Adjust()):
    """Apply look to display-referred sRGB values (..., 3) in 0..1."""
    x = np.clip(np.asarray(display, dtype=np.float64), 0.0, 1.0)
    out = _white_balance(x, look.temp + adjust.temp * 0.3, look.tint + adjust.tint * 0.3)
    if look.mono:
        w = np.asarray(look.mono, dtype=np.float64)
        y = srgb_to_linear(out) @ (w / w.sum())
        out = np.repeat(linear_to_srgb(y)[..., None], 3, axis=-1)
    out = np.clip(_tone(out, look, adjust), 0.0, 1.0)
    lab = _colour(linear_to_oklab(srgb_to_linear(out)), look, adjust)
    out = linear_to_srgb(_gamut_map(oklab_to_linear(lab), lab[..., 0]))
    return np.clip(x + adjust.strength * (out - x), 0.0, 1.0)


def bake(look, adjust=Adjust(), grid=33):
    return render(look, identity(grid), adjust)


NEUTRAL = Look("neutral", "原样", "")


def bake_lut(table, adjust=Adjust(), grid=33):
    """An imported LUT with the user's fine-tuning applied after it."""
    x = identity(grid)
    graded = render(NEUTRAL, apply(table, x), replace(adjust, strength=1.0))
    return x + adjust.strength * (graded - x)


# ---------------------------------------------------------------- presets

HNCS = "哈苏 HNCS 风格（近似）"
FUJI = "富士胶片模拟风格（近似）"
MONO = "黑白"

PRESETS = [
    Look(
        "hncs", "HNCS 自然", HNCS,
        "自然还原、肤色通透、高光过渡柔和、饱和度克制。",
        curve=((0, 0.008), (0.1, 0.095), (0.5, 0.5), (0.85, 0.86), (0.96, 0.945), (1, 0.98)),
        curve_luma=0.6, temp=0.025, saturation=0.97, vibrance=0.06, highlight_sat=0.9,
        bands=(("red", 2, 0.95, 0.0), ("orange", 2, 1.02, 0.012), ("yellow", -3, 0.92, 0.0),
               ("green", -6, 0.88, -0.01), ("cyan", -4, 0.92, 0.0), ("blue", -4, 0.95, -0.01)),
    ),
    Look(
        "hncs_portrait", "HNCS 人像", HNCS,
        "更低反差、更亮更干净的肤色，适合人像。",
        curve=((0, 0.015), (0.1, 0.105), (0.5, 0.51), (0.85, 0.855), (1, 0.975)),
        curve_luma=0.7, temp=0.03, saturation=0.94, vibrance=0.04, highlight_sat=0.88,
        bands=(("red", 3, 0.92, 0.0), ("orange", 3, 0.97, 0.025), ("yellow", -2, 0.9, 0.01),
               ("green", -6, 0.85, 0.0), ("blue", -3, 0.95, 0.0), ("magenta", -4, 0.9, 0.0)),
    ),
    Look(
        "hncs_landscape", "HNCS 风光", HNCS,
        "层次更深、蓝绿更沉稳，保持自然不过饱和。",
        curve=((0, 0.0), (0.1, 0.085), (0.3, 0.28), (0.5, 0.5), (0.75, 0.77), (0.95, 0.94), (1, 0.98)),
        curve_luma=0.5, temp=0.015, saturation=1.02, vibrance=0.1, highlight_sat=0.92,
        bands=(("yellow", -3, 0.95, 0.0), ("green", -5, 0.95, -0.02), ("cyan", -3, 1.0, -0.015),
               ("blue", -3, 1.05, -0.03), ("orange", 2, 1.0, 0.01)),
    ),
    Look(
        "provia", "PROVIA 标准", FUJI,
        "均衡的反差与饱和度，适合大多数场景。",
        curve=((0, 0), (0.12, 0.1), (0.5, 0.5), (0.85, 0.875), (1, 1)),
        saturation=1.08,
        bands=(("red", -2, 1.05, 0.0), ("green", -3, 1.05, 0.0), ("blue", -2, 1.06, -0.01)),
    ),
    Look(
        "velvia", "Velvia 鲜艳", FUJI,
        "高饱和、高反差，蓝天与绿植浓郁，适合风光。",
        curve=((0, 0), (0.12, 0.085), (0.3, 0.27), (0.5, 0.5), (0.75, 0.79), (0.9, 0.93), (1, 1)),
        saturation=1.2, vibrance=0.1, shadow_sat=1.06,
        bands=(("red", -2, 1.08, -0.01), ("orange", -1, 1.05, 0.0), ("yellow", -3, 1.08, 0.0),
               ("green", -4, 1.15, -0.02), ("cyan", -2, 1.12, -0.02), ("blue", 0, 1.12, -0.03),
               ("magenta", 0, 1.06, 0.0)),
    ),
    Look(
        "astia", "ASTIA 柔和", FUJI,
        "柔和的反差、明亮的肤色与天空，人像友好。",
        curve=((0, 0.01), (0.12, 0.115), (0.5, 0.505), (0.85, 0.86), (1, 0.99)),
        saturation=1.05,
        bands=(("orange", 0, 0.95, 0.02), ("red", -1, 1.0, 0.01), ("green", -3, 1.08, 0.0),
               ("blue", -3, 1.1, 0.02), ("cyan", -2, 1.05, 0.01)),
    ),
    Look(
        "classic_chrome", "Classic Chrome 经典正片", FUJI,
        "低饱和、硬朗的暗部，青调的蓝、偏橙的红，纪实感。",
        curve=((0, 0), (0.1, 0.07), (0.25, 0.205), (0.5, 0.48), (0.75, 0.76), (0.92, 0.91), (1, 0.97)),
        saturation=0.8, highlight_sat=0.85,
        bands=(("red", 6, 0.8, -0.01), ("orange", 3, 0.88, 0.0), ("yellow", -6, 0.75, -0.01),
               ("green", -10, 0.7, -0.02), ("cyan", -8, 0.82, 0.0), ("blue", -10, 0.78, -0.03),
               ("magenta", -4, 0.8, 0.0)),
        shadow_tint=(220, 0.006), highlight_tint=(70, 0.006),
    ),
    Look(
        "classic_neg", "Classic Neg 经典负片", FUJI,
        "硬调中间调、暗部青绿、高光偏暖，红色偏洋红，街拍胶片感。",
        curve=((0, 0.01), (0.12, 0.085), (0.3, 0.27), (0.5, 0.5), (0.72, 0.75), (0.9, 0.9), (1, 0.97)),
        saturation=0.9, temp=-0.01,
        bands=(("red", -6, 1.05, 0.0), ("orange", -4, 0.9, 0.0), ("yellow", -8, 0.8, 0.0),
               ("green", 12, 0.75, -0.02), ("cyan", 0, 0.9, 0.0), ("blue", -6, 0.9, -0.01)),
        shadow_tint=(190, 0.02), highlight_tint=(45, 0.012),
    ),
    Look(
        "nostalgic_neg", "Nostalgic Neg 怀旧负片", FUJI,
        "琥珀色高光、柔和暗部，七十年代彩色摄影的温暖。",
        curve=((0, 0.02), (0.15, 0.16), (0.5, 0.51), (0.85, 0.85), (1, 0.97)),
        saturation=0.95, temp=0.03,
        bands=(("red", 6, 1.0, 0.0), ("orange", 2, 1.05, 0.01), ("yellow", -4, 0.95, 0.0),
               ("green", 6, 0.85, 0.0), ("blue", -4, 0.9, 0.0)),
        highlight_tint=(75, 0.018),
    ),
    Look(
        "eterna", "ETERNA 电影", FUJI,
        "低反差、低饱和、暗部带青，宽容度高的电影感。",
        curve=((0, 0.03), (0.15, 0.16), (0.5, 0.49), (0.85, 0.82), (1, 0.95)),
        saturation=0.72,
        bands=(("orange", 0, 0.88, 0.0), ("yellow", -4, 0.75, 0.0), ("green", 8, 0.65, -0.01),
               ("cyan", 0, 0.85, 0.0), ("blue", -6, 0.8, 0.0)),
        shadow_tint=(210, 0.012),
    ),
    Look(
        "bleach_bypass", "ETERNA 漂白", FUJI,
        "极低饱和、高反差的漂白效果。",
        curve=((0, 0), (0.12, 0.07), (0.3, 0.25), (0.5, 0.5), (0.7, 0.75), (0.9, 0.94), (1, 1)),
        saturation=0.42, temp=-0.01,
    ),
    Look(
        "pro_neg_std", "PRO Neg. Std 人像负片", FUJI,
        "柔和平顺、肤色自然，适合棚拍人像。",
        curve=((0, 0.01), (0.15, 0.15), (0.5, 0.505), (0.85, 0.845), (1, 0.98)),
        curve_luma=0.4, saturation=0.9,
        bands=(("orange", 0, 0.95, 0.015), ("red", 0, 0.95, 0.0), ("green", -2, 0.9, 0.0)),
    ),
    Look(
        "reala_ace", "REALA ACE 真实", FUJI,
        "忠实还原色彩，硬朗而干净的影调。",
        curve=((0, 0), (0.12, 0.105), (0.5, 0.5), (0.85, 0.865), (1, 0.995)),
        saturation=1.0, vibrance=0.04,
        bands=(("orange", 0, 1.0, 0.008), ("green", -2, 0.98, 0.0), ("blue", -2, 1.02, -0.01)),
    ),
    Look(
        "acros", "ACROS 黑白", MONO,
        "细腻颗粒感的高解析黑白，暗部深沉、高光细腻。",
        mono=(0.27, 0.62, 0.11),
        curve=((0, 0), (0.1, 0.07), (0.3, 0.26), (0.5, 0.5), (0.75, 0.78), (0.92, 0.93), (1, 1)),
    ),
    Look(
        "acros_r", "ACROS + 红滤镜", MONO,
        "压暗蓝天、提亮肤色与红色。",
        mono=(0.62, 0.33, 0.05),
        curve=((0, 0), (0.1, 0.07), (0.3, 0.26), (0.5, 0.5), (0.75, 0.78), (0.92, 0.93), (1, 1)),
    ),
    Look(
        "mono", "单色 标准", MONO, "标准亮度权重的黑白。",
        mono=tuple(LUMA), curve=((0, 0), (0.15, 0.13), (0.5, 0.5), (0.85, 0.87), (1, 1)),
    ),
    Look(
        "sepia", "棕褐色", MONO, "暖棕色调的单色。",
        mono=tuple(LUMA), curve=((0, 0.02), (0.5, 0.5), (1, 0.97)),
        shadow_tint=(65, 0.035), highlight_tint=(75, 0.03),
    ),
]

BY_KEY = {look.key: look for look in PRESETS}

