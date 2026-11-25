"""Render Umeda views on Kaggle's free GPUs (T4 x2) instead of the local CPU.

  python3 scripts/kaggle/kaggle_render.py bundle            # upload scripts/fonts/textures as a PRIVATE dataset
  python3 scripts/kaggle/kaggle_render.py run arch gate ... [--samples 512 --noise 0.008 --width 1920]
                                                             # push a GPU notebook, wait, download the EXRs to render/

Needs the Kaggle CLI (pip install kaggle) and an API token (KAGGLE_API_TOKEN or ~/.kaggle).
The official station map and the reference photos are NOT uploaded: rendering only needs
the derived plan.json, the scripts, the fonts and the texture caches.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.path.join(ROOT, "cache", "kaggle")
DATASET = "umeda-render-assets"
KERNEL = "umeda-gpu-render"

# texture caches the current materials load (anything missing is regenerated on Kaggle)
CACHES = [
    "stone_b1_floor2_2048", "pavers_b2s_2048", "pavers_b2bands_2048", "wall_umeda_2048",
    "panel_b1_wall_2048", "panel_vault_perf_2048", "panel_beige_wall_2048", "panel_ceil_b2b_2048",
    "panel_parapet_2048", "panel_esc_step_1024", "panel_vault_metal2_2048",
    "glazed_track_tile2_2048", "glazed_wall_white_2048", "stone_stair_granite_1024",
    "tactile_dots_1024", "tactile_bars_1024", "checker_1024", "vault_tex_125",
]


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def user():
    p = os.path.expanduser("~/.kaggle/kaggle.json")
    return json.load(open(p))["username"]


def bundle():
    d = os.path.join(WORK, "dataset")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "proj", "cache", "umeda"))
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(d, "proj", "scripts"),
                    ignore=shutil.ignore_patterns("__pycache__", "kaggle"))
    shutil.copytree(os.path.join(ROOT, "fonts"), os.path.join(d, "proj", "fonts"),
                    ignore=shutil.ignore_patterns("*.sh", "*.md"))
    shutil.copy(os.path.join(ROOT, "cache", "umeda", "plan.json"), os.path.join(d, "proj", "cache", "umeda"))
    for c in CACHES:
        src = os.path.join(ROOT, "cache", c + ".npz")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, "proj", "cache"))
        else:
            print("not cached locally (Kaggle will generate it):", c)
    # one tar so the folder layout survives the upload unchanged
    sh(["tar", "-cf", os.path.join(d, "proj.tar"), "-C", d, "proj"])
    shutil.rmtree(os.path.join(d, "proj"))
    meta = {"title": DATASET, "id": f"{user()}/{DATASET}", "licenses": [{"name": "other"}]}
    json.dump(meta, open(os.path.join(d, "dataset-metadata.json"), "w"))
    r = subprocess.run(["kaggle", "datasets", "status", f"{user()}/{DATASET}"], capture_output=True, text=True)
    if "ready" in r.stdout or "pending" in r.stdout:
        sh(["kaggle", "datasets", "version", "-p", d, "-m", "update", "-q"])
    else:
        sh(["kaggle", "datasets", "create", "-p", d, "-q"])        # private unless --public is given


KERNEL_CODE = r'''
import glob, os, subprocess, sys, time, tarfile
VIEWS = __VIEWS__
ARGS = __ARGS__
t0 = time.time()
def run(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)
tars = glob.glob("/kaggle/input/**/proj.tar", recursive=True)
src = glob.glob("/kaggle/input/**/proj/scripts/umeda/build_umeda.py", recursive=True)
os.makedirs("/tmp/w", exist_ok=True)
if tars:
    tarfile.open(tars[0]).extractall("/tmp/w")
else:
    run(f"cp -r {os.path.dirname(os.path.dirname(os.path.dirname(src[0])))} /tmp/w/proj")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip -q install uv")
run("uv venv -q /tmp/venv -p 3.13")
run("uv pip install -q --python /tmp/venv/bin/python bpy==5.3.0 numpy scipy pillow opencv-python-headless")
try:
    run("/tmp/venv/bin/python -c 'import bpy'")
except subprocess.CalledProcessError:
    run("apt-get -qq update && apt-get -qq install -y libxi6 libxkbcommon0 libxxf86vm1 libsm6 libgl1 libxrender1 libxfixes3 > /dev/null")
print(f"setup {time.time() - t0:.0f}s", flush=True)
env = dict(os.environ, UMEDA_GPU="1")
for v in VIEWS:
    t1 = time.time()
    run(f"/tmp/venv/bin/python build_umeda.py --view {v} --out /kaggle/working/umeda_{v}.exr {ARGS}",
        cwd="/tmp/w/proj/scripts/umeda", env=env)
    print(f"view {v} {time.time() - t1:.0f}s", flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
'''


def run(views, args):
    d = os.path.join(WORK, "kernel")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    extra = f"--width {args.width} --samples {args.samples} --noise {args.noise}"
    code = KERNEL_CODE.replace("__VIEWS__", repr(views)).replace("__ARGS__", repr(extra))
    open(os.path.join(d, "run.py"), "w").write(code)
    meta = {"id": f"{user()}/{KERNEL}", "title": KERNEL, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{user()}/{DATASET}"],
            "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"))
    sh(["kaggle", "kernels", "push", "-p", d])
    ref = f"{user()}/{KERNEL}"
    t0 = time.time()
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
        st = r.stdout.strip()
        print(f"[{time.time() - t0:5.0f}s] {st}", flush=True)
        low = st.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            break
    out = os.path.join(WORK, "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    sh(["kaggle", "kernels", "output", ref, "-p", out])
    got = []
    for f in os.listdir(out):
        if f.endswith(".exr"):
            shutil.move(os.path.join(out, f), os.path.join(ROOT, "render", f))
            got.append(f)
    print("downloaded:", got)
    log = [f for f in os.listdir(out) if f.endswith(".log")]
    for f in log:
        print(open(os.path.join(out, f)).read()[-3000:])
    if "complete" not in st.lower() or len(got) < len(views):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bundle", "run"])
    ap.add_argument("views", nargs="*")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--noise", type=float, default=0.008)
    a = ap.parse_args()
    if a.cmd == "bundle":
        bundle()
    else:
        run(a.views, a)


if __name__ == "__main__":
    main()
