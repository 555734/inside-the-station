"""Lens / camera post-processing on the linear Cycles EXR, then AgX tone mapping via Blender.

usage: python3 post.py in.exr out.png [--exposure 0] [--look "AgX - Base Contrast"]
"""
import argparse
import os

import bpy
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def load_exr(path):
    img = bpy.data.images.load(os.path.abspath(path))
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)[..., :3].copy()


def bloom(a, scale, thr, strength):
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    k = np.clip(lum - thr, 0, None) / (lum + 1e-6)
    bright = a * k[..., None]
    out = np.zeros_like(a)
    for sigma, wgt in ((2.0, 0.35), (6.0, 0.30), (18.0, 0.22), (55.0, 0.13)):
        for c in range(3):
            out[..., c] += wgt * ndimage.gaussian_filter(bright[..., c], sigma * scale, mode="nearest")
    return a + strength * out


def veil(a, scale, strength):
    """Very wide, low-level glare (lens veiling) that lifts the blacks around bright areas."""
    small = a[::8, ::8]
    g = np.stack([ndimage.gaussian_filter(small[..., c], 40 * scale / 8, mode="nearest") for c in range(3)], -1)
    g = ndimage.zoom(g, (a.shape[0] / g.shape[0], a.shape[1] / g.shape[1], 1), order=1)
    return a + strength * g[: a.shape[0], : a.shape[1]]


def chromatic(a, amount):
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    out = a.copy()
    for c, s in ((0, 1 + amount), (2, 1 - amount)):
        xs = cx + (xx - cx) / s
        ys = cy + (yy - cy) / s
        out[..., c] = ndimage.map_coordinates(a[..., c], [ys, xs], order=1, mode="nearest")
    return out


def vignette(a, strength):
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.75 + ((yy - h / 2) / (w / 2)) ** 2 * 0.75
    return a * (1 - strength * r2)[..., None]


def tonemap_with_blender(a, path, exposure, look, view="AgX"):
    h, w, _ = a.shape
    rgba = np.concatenate([a, np.ones((h, w, 1), np.float32)], -1)
    img = bpy.data.images.new("post", w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.pixels.foreach_set(rgba.ravel())
    sc = bpy.context.scene
    sc.view_settings.view_transform = view
    try:
        if look and look != "None":
            sc.view_settings.look = look
    except TypeError as e:
        print("look not available:", e)
    sc.view_settings.exposure = exposure
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    img.save_render(os.path.abspath(path), scene=sc)


def grain(path_in, path_out, amount, seed=7):
    im = Image.open(path_in).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, a.shape[:2]).astype(np.float32)
    n = ndimage.gaussian_filter(n, 0.6)
    n /= n.std()
    lum = a.mean(-1, keepdims=True) / 255
    a += n[..., None] * amount * (0.6 + 0.4 * (1 - lum)) * 255 / 100
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=35, threshold=2))
    im.save(path_out)
    if path_out.endswith(".png"):
        im.save(path_out[:-4] + ".jpg", quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--look", default="AgX - Base Contrast")
    ap.add_argument("--view", default="AgX")
    ap.add_argument("--bloom", type=float, default=0.06)
    ap.add_argument("--thr", type=float, default=2.0)
    ap.add_argument("--veil", type=float, default=0.015)
    ap.add_argument("--ca", type=float, default=0.0012)
    ap.add_argument("--vignette", type=float, default=0.22)
    ap.add_argument("--grain", type=float, default=1.6)
    args = ap.parse_args()
    a = load_exr(args.inp)
    a *= 2 ** args.exposure
    scale = a.shape[1] / 1920
    a = bloom(a, scale, args.thr, args.bloom)
    a = veil(a, scale, args.veil)
    a = chromatic(a, args.ca)
    a = vignette(a, args.vignette)
    tmp = args.out + ".tmp.png"
    tonemap_with_blender(a, tmp, 0.0, args.look, args.view)
    grain(tmp, args.out, args.grain)
    os.remove(tmp)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
