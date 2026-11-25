"""Turn the raw baked lightmaps into the app's lightmap PNGs (runs locally, needs no GPU).

  python3 finish_station.py app/station cache/kaggle/lm_raw [more raw dirs...] [--no-denoise]

For every chunk listed in OUT/app.json the raw float16 irradiance (RAW/<chunk>.npz, written by
export_station.py) is denoised with Open Image Denoise and stored as OUT/lightmaps/<chunk>.png
(sqrt(E / (E + 1)) in 8 bits, the encoding the app decodes). Stops with a list of the chunks
that have no raw lightmap yet.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--no-denoise", action="store_true")
    a = ap.parse_args()
    meta_p = os.path.join(a.out, "app.json")
    meta = json.load(open(meta_p))
    os.makedirs(os.path.join(a.out, "lightmaps"), exist_ok=True)
    missing = []
    for name, info in sorted(meta["lightmaps"].items()):
        src = next((os.path.join(d, name + ".npz") for d in a.raw if os.path.exists(os.path.join(d, name + ".npz"))), None)
        if src is None:
            missing.append(name)
            continue
        e = np.load(src)["lm"].astype(np.float32)
        if not a.no_denoise:
            e = denoise(e)
        info["p99"] = save_lightmap(e, os.path.join(a.out, info["file"]))
        print(name, e.shape[1], "px", "denoised" if not a.no_denoise else "raw", flush=True)
    json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=1)
    if missing:
        print("no raw lightmap yet for", len(missing), "chunks:", ",".join(missing))
        sys.exit(2)
    print("all", len(meta["lightmaps"]), "lightmaps written")


if __name__ == "__main__":
    main()
