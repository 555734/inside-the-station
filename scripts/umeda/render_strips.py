"""Restart-safe rendering: render a view as horizontal strips (each its own Blender run, saved
as soon as it is done), skip strips that already exist, then stitch them into one EXR.

usage: python3 render_strips.py VIEW OUT.exr [--strips 6] [--width 1920] [--samples 160] [--noise 0.015]
Strips overlap by a few pixels and are cross-faded, so per-strip denoising leaves no seam.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stitch(parts, out, W, H, ov):
    import bpy
    import numpy as np
    acc = np.zeros((H, W, 4), np.float32)
    wsum = np.zeros((H, 1, 1), np.float32)
    for (y0, y1, path) in parts:              # y measured from the bottom (Blender convention)
        img = bpy.data.images.load(os.path.abspath(path))
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)
        assert w == W and h == y1 - y0, (path, w, h, y0, y1)
        wt = np.ones((h, 1, 1), np.float32)
        r = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1, None, None]
        if y0 > 0:
            wt[:ov] = r
        if y1 < H:
            wt[-ov:] = r[::-1]
        acc[y0:y1] += a * wt
        wsum[y0:y1] += wt
        bpy.data.images.remove(img)
    acc /= np.maximum(wsum, 1e-6)
    img = bpy.data.images.new("stitched", W, H, alpha=True, float_buffer=True)
    img.pixels.foreach_set(acc.ravel())
    img.filepath_raw = os.path.abspath(out)
    img.file_format = "OPEN_EXR"
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    img.save_render(os.path.abspath(out), scene=sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view")
    ap.add_argument("out")
    ap.add_argument("--strips", type=int, default=6)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--overlap", type=int, default=24)
    args = ap.parse_args()
    W = args.width
    H = int(round(W * 9 / 16))
    n, ov = args.strips, args.overlap
    base = os.path.abspath(args.out)[:-4]
    parts = []
    for i in range(n):
        y0 = max(0, round(H * i / n) - ov // 2)
        y1 = min(H, round(H * (i + 1) / n) + ov // 2)
        path = f"{base}_strip{i}.exr"
        parts.append((y0, y1, path))
        if os.path.exists(path):
            print("skip", path, flush=True)
            continue
        tmp = f"{base}_strip{i}_tmp.exr"
        # Blender truncates border*height to an int: aim at the pixel centre so it lands exactly
        border = f"0,{(y0 + 0.5) / H:.7f},1,{(y1 + 0.5) / H if y1 < H else 1.0:.7f}"
        cmd = [sys.executable, os.path.join(HERE, "build_umeda.py"), "--view", args.view, "--out", tmp,
               "--width", str(W), "--samples", str(args.samples), "--noise", str(args.noise), "--border", border]
        subprocess.run(cmd, check=True, cwd=HERE)
        os.replace(tmp, path)           # a strip only counts once it is complete
        print("strip done", i, flush=True)
    stitch(parts, args.out, W, H, ov)
    print("stitched", args.out, flush=True)


if __name__ == "__main__":
    main()
