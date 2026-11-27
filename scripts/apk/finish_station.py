"""Turn the raw baked lightmaps into the app's lightmap PNGs (runs locally, needs no GPU).

  python3 finish_station.py app/station cache/kaggle/lm_raw [more raw dirs...] [--no-denoise] [--resume]

For every chunk listed in OUT/app.json the raw float16 irradiance (RAW/<chunk>.npz, written by
export_station.py) is denoised with Open Image Denoise and stored as OUT/lightmaps/<chunk>.png
(sqrt(E / (E + 1)) in 8 bits, the encoding the app decodes), and a 1/8 size stand-in as
OUT/lightmaps_lo/<chunk>.png (shown while the app has the full one unloaded). Stops with a list of the chunks
that have no raw lightmap yet. With --resume, chunks whose PNG is already newer than their raw
lightmap are left as they are (so it can be run while bakes are still coming in).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_denoise import denoise  # noqa: E402


def save_lightmap(e, path):
    e = np.clip(np.asarray(e, np.float32), 0, None)
    enc = np.sqrt(e / (e + 1.0))
    Image.fromarray((enc[::-1] * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(path, optimize=True)
    return float(np.percentile(e, 99))


LO = 8

IMPORT = """[remap]

importer="texture"
type="CompressedTexture2D"

[params]

compress/mode=2
compress/high_quality=true
mipmaps/generate=true
"""


def write_import(png):
    """Godot import settings for a lightmap: VRAM compressed in high quality (ASTC 4x4 on phones,
    BPTC on desktops). The default (ETC2 on phones) shifted the colour of whole 4x4 blocks of a
    near-white lightmap, which showed as pink and green blotches on the floors."""
    p = png + ".import"
    if os.path.exists(p) and "compress/high_quality=true" in open(p).read():
        return
    open(p, "w").write(IMPORT)


# Godot import settings of the models. Nothing in the app casts a real-time shadow (the light is all
# baked), so the shadow meshes Godot makes by default are never drawn; the collision model only gives
# its triangles to the physics (no levels of detail, tangents or shadow meshes needed). Leaving them
# out keeps the APK under 400 MB with the same picture.
SCENE_PARAMS = {
    "station.glb": {"meshes/create_shadow_meshes": "false"},
    "far.glb": {"meshes/create_shadow_meshes": "false"},
    "collision.glb": {"meshes/create_shadow_meshes": "false", "meshes/generate_lods": "false",
                      "meshes/ensure_tangents": "false"},
}


def write_scene_imports(out):
    """Import settings of the station's models (see SCENE_PARAMS); a model whose settings change is
    imported again by the next `godot --import`."""
    import glob
    import re
    for f, params in SCENE_PARAMS.items():
        if not os.path.exists(os.path.join(out, f)):
            continue
        p = os.path.join(out, f + ".import")
        old = open(p).read() if os.path.exists(p) else ""
        txt = old or '[remap]\n\nimporter="scene"\nimporter_version=1\n\n[params]\n\n'
        for k, v in params.items():
            if re.search(rf"^{re.escape(k)}=", txt, re.M):
                txt = re.sub(rf"^{re.escape(k)}=.*$", f"{k}={v}", txt, flags=re.M)
            else:
                txt = txt.replace("[params]\n", f"[params]\n{k}={v}\n", 1)
        if txt != old:
            open(p, "w").write(txt)
            for m in glob.glob(os.path.join(os.path.dirname(os.path.abspath(out)), ".godot", "imported", f + "-*.md5")):
                os.remove(m)


def save_lowres(e, path):
    """A 1/8 size stand-in the app shows while the full lightmap of a chunk is not loaded: mean of
    the covered texels in each 8 x 8 block (empty gutter texels left out), empty blocks filled
    from the nearest covered one."""
    from scipy import ndimage
    e = np.clip(np.asarray(e, np.float32), 0, None)
    h, w = e.shape[0] // LO * LO, e.shape[1] // LO * LO
    e = e[:h, :w]
    m = (e.sum(-1) > 1e-6).astype(np.float32)
    s = (e * m[..., None]).reshape(h // LO, LO, w // LO, LO, 3).sum((1, 3))
    c = m.reshape(h // LO, LO, w // LO, LO).sum((1, 3))
    lo = s / np.maximum(c, 1.0)[..., None]
    empty = c == 0
    if empty.any() and not empty.all():
        idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        lo = lo[idx[0], idx[1]]
    save_lightmap(lo, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    meta_p = os.path.join(a.out, "app.json")
    meta = json.load(open(meta_p))
    write_scene_imports(a.out)
    os.makedirs(os.path.join(a.out, "lightmaps"), exist_ok=True)
    missing = []
    for name, info in sorted(meta["lightmaps"].items()):
        src = next((os.path.join(d, name + ".npz") for d in a.raw if os.path.exists(os.path.join(d, name + ".npz"))), None)
        if src is None:
            missing.append(name)
            continue
        dst = os.path.join(a.out, info["file"])
        if a.resume and "p99" in info and os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
            continue
        e = np.load(src)["lm"].astype(np.float32)
        if not a.no_denoise:
            e = denoise(e)
        info["p99"] = save_lightmap(e, dst)
        write_import(dst)
        lo_dst = os.path.join(a.out, "lightmaps_lo", os.path.basename(info["file"]))
        os.makedirs(os.path.dirname(lo_dst), exist_ok=True)
        save_lowres(e, lo_dst)
        write_import(lo_dst)
        info["lo"] = os.path.relpath(lo_dst, a.out)
        print(name, e.shape[1], "px", "denoised" if not a.no_denoise else "raw", flush=True)
        json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=1)
    # the unbaked far part of the avenue is lit like the baked street near by
    # (every 4th texel each way is plenty for a median, and keeps the memory small)
    up = [np.load(next(os.path.join(d, n + ".npz") for d in a.raw if os.path.exists(os.path.join(d, n + ".npz"))))["lm"][::4, ::4]
          for n in meta["lightmaps"] if n.startswith("UP_") and n not in missing]
    if up:
        v = np.concatenate([u.astype(np.float32).mean(-1).ravel() for u in up])
        v = v[v > 1e-4]
        meta["far_irradiance"] = float(np.median(v)) if v.size else 3.0
    json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=1)
    if missing:
        print("no raw lightmap yet for", len(missing), "chunks:", ",".join(missing))
        sys.exit(2)
    print("all", len(meta["lightmaps"]), "lightmaps written")


if __name__ == "__main__":
    main()
