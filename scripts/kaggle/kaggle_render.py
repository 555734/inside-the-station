"""Render Umeda views on Kaggle's free GPUs (T4 x2) instead of the local CPU.

  python3 scripts/kaggle/kaggle_render.py bundle            # upload scripts/fonts/textures as a PRIVATE dataset
  python3 scripts/kaggle/kaggle_render.py run arch gate ... [--samples 512 --noise 0.008 --width 1920]
                                                             # push a GPU notebook, wait, download the EXRs to render/
  python3 scripts/kaggle/kaggle_render.py run __export__ --samples 128
                                                             # export the walkthrough data into app/station/ and bake
                                                             # as many lightmaps as the GPU quota left this week allows
                                                             # (raw ones collect in cache/kaggle/lm_raw/; run again to
                                                             # continue, then scripts/apk/finish_station.py)

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


def layout_id(uv_dir):
    p = os.path.join(uv_dir, "index.json")
    return json.load(open(p)).get("layout_id", "") if os.path.exists(p) else ""


def bundle():
    d = os.path.join(WORK, "dataset")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "proj", "cache", "umeda"))
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(d, "proj", "scripts"),
                    ignore=shutil.ignore_patterns("__pycache__", "kaggle"))
    shutil.copytree(os.path.join(ROOT, "fonts"), os.path.join(d, "proj", "fonts"),
                    ignore=shutil.ignore_patterns("*.sh", "*.md"))
    shutil.copy(os.path.join(ROOT, "cache", "umeda", "plan.json"), os.path.join(d, "proj", "cache", "umeda"))
    uv = os.path.join(WORK, "apk_uv")
    if os.path.isdir(uv):          # lightmap layouts (made locally or by an earlier run) the bake must use
        shutil.copytree(uv, os.path.join(d, "proj", "cache", "apk_uv"))
        open(os.path.join(WORK, "bundled_layout"), "w").write(layout_id(uv))
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
BUDGET = __BUDGET__
SKIP = __SKIP__
LAYOUT = __LAYOUT__
DEVICE = __DEVICE__
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
import json
if LAYOUT:
    ip = "/tmp/w/proj/cache/apk_uv/index.json"
    got = json.load(open(ip)).get("layout_id") if os.path.exists(ip) else None
    if got != LAYOUT:
        sys.exit(f"dataset holds layout {got}, expected {LAYOUT}: the new dataset version was not ready")
# the scripts travel with the notebook (always current); the dataset only carries fonts/textures
import base64, io
tarfile.open(fileobj=io.BytesIO(base64.b64decode(SCRIPTS_B64))).extractall("/tmp/w/proj")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip -q install uv")
run("uv venv -q /tmp/venv -p 3.13")
run("uv pip install -q --python /tmp/venv/bin/python bpy==5.3.0 numpy scipy pillow opencv-python-headless")
try:
    run("/tmp/venv/bin/python -c 'import bpy'")
except subprocess.CalledProcessError:
    run("apt-get -qq update && apt-get -qq install -y libxi6 libxkbcommon0 libxxf86vm1 libsm6 libgl1 libxrender1 libxfixes3 > /dev/null")
print(f"setup {time.time() - t0:.0f}s", flush=True)
env = dict(os.environ, UMEDA_GPU="1") if DEVICE == "gpu" else dict(os.environ)
if VIEWS == ["__export__"]:
    t1 = time.time()
    resume = ""
    if os.path.exists("/tmp/w/proj/cache/apk_uv/index.json"):
        # the layouts made and checked locally (export_station.py --layout-only): same chunks,
        # face orientation, closures and lightmap UVs, so every run bakes the same lightmaps
        resume = "--uv-from /tmp/w/proj/cache/apk_uv"
        if SKIP:
            resume += " --skip " + ",".join(SKIP)
    # raw lightmaps go straight to /kaggle/working as each batch finishes; no batch starts that
    # could not end before the GPU quota left this week runs out
    dev = "--gpu" if DEVICE == "gpu" else "--reverse"      # TPU machine: bake on its CPUs, from the other end
    run(f"/tmp/venv/bin/python export_station.py /tmp/station {ARGS} {dev} --raw-dir /kaggle/working "
        f"--zip /kaggle/working/station.zip --deadline {t0 + BUDGET:.0f} {resume}",
        cwd="/tmp/w/proj/scripts/apk", env=env)
    print(f"export {time.time() - t1:.0f}s", flush=True)
    VIEWS = []
for v in VIEWS:
    t1 = time.time()
    run(f"/tmp/venv/bin/python build_umeda.py --view {v} --out /kaggle/working/umeda_{v}.exr {ARGS}",
        cwd="/tmp/w/proj/scripts/umeda", env=env)
    print(f"view {v} {time.time() - t1:.0f}s", flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
'''


def gpu_quota_left(kind="gpu"):
    """Seconds of Kaggle GPU (or TPU) time left this week (the session is stopped when it runs out)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    import kagglesdk.kernels.types.kernels_api_service as t
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        q = k.kernels.kernels_api_client.get_accelerator_quota_statistics(
            t.ApiGetAcceleratorQuotaStatisticsRequest()).to_dict()[kind + "Quota"]
    sec = lambda v: float(str(v).rstrip("s").split(".")[0] or 0)      # noqa: E731 ("11662.902555.0s")
    left = sec(q["totalTimeAllowed"]) - sec(q["timeUsed"])
    print(f"{kind.upper()} quota: {sec(q['timeUsed']) / 3600:.2f} h used of {sec(q['totalTimeAllowed']) / 3600:.1f} h, "
          f"refresh {q.get('quotaRefreshTime')}", flush=True)
    return left


def run(views, args):
    d = os.path.join(WORK, "kernel")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    if views == ["__export__"]:
        extra = f"--samples {args.samples} --texel {args.texel} --max-lm {args.max_lm} --max-tex {args.max_tex}"
        if args.no_cull:
            extra += " --no-cull"
    else:
        extra = f"--width {args.width} --samples {args.samples} --noise {args.noise}"
    import base64
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(os.path.join(ROOT, "scripts"), arcname="scripts",
               filter=lambda ti: None if ("__pycache__" in ti.name or "/kaggle" in ti.name) else ti)
    b64 = base64.b64encode(buf.getvalue()).decode()
    budget, skip = 0, []
    if views == ["__export__"]:
        budget = gpu_quota_left("tpu" if args.tpu else "gpu") - args.margin
        budget = min(budget, 9 * 3600 - args.margin)          # one session runs at most 9 hours
        if budget < 1800:
            sys.exit(f"only {budget + args.margin:.0f}s of GPU quota left this week")
        raw = os.path.join(WORK, "lm_raw")
        lid = layout_id(os.path.join(WORK, "apk_uv"))
        if lid:
            bp = os.path.join(WORK, "bundled_layout")
            if not os.path.exists(bp) or open(bp).read().strip() != lid:
                sys.exit("cache/kaggle/apk_uv changed since the last upload: run the bundle command first")
        if os.path.isdir(raw):
            skip = sorted(f[:-4] for f in os.listdir(raw) if f.endswith(".npz"))
        if skip:
            lp = os.path.join(raw, "LAYOUT")
            if not lid or not os.path.exists(lp) or open(lp).read().strip() != lid:
                sys.exit("cache/kaggle/lm_raw holds lightmaps of another layout: move them away first")
        print(f"bake budget {budget:.0f}s, {len(skip)} chunks already baked", flush=True)
    code = KERNEL_CODE.replace("__VIEWS__", repr(views)).replace("__ARGS__", repr(extra))
    code = code.replace("__BUDGET__", repr(budget)).replace("__SKIP__", repr(skip))
    code = code.replace("__DEVICE__", repr("tpu" if args.tpu else "gpu"))
    code = code.replace("__LAYOUT__", repr(layout_id(os.path.join(WORK, "apk_uv")) if views == ["__export__"] else ""))
    code = "SCRIPTS_B64 = " + repr(b64) + "\n" + code
    open(os.path.join(d, "run.py"), "w").write(code)
    kernel = KERNEL + ("-tpu" if args.tpu else "")
    meta = {"id": f"{user()}/{kernel}", "title": kernel, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_internet": "true",
            "enable_gpu": "false" if args.tpu else "true", "enable_tpu": "true" if args.tpu else "false",
            "machine_shape": "TpuV5E8" if args.tpu else "NvidiaTeslaT4", "dataset_sources": [f"{user()}/{DATASET}"],
            "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"))
    sh(["kaggle", "kernels", "push", "-p", d])
    ref = f"{user()}/{kernel}"
    t0 = time.time()
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
        st = r.stdout.strip()
        print(f"[{time.time() - t0:5.0f}s] {st}", flush=True)
        low = st.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            break
    out = os.path.join(WORK, "output_tpu" if args.tpu else "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    sh(["kaggle", "kernels", "output", ref, "-p", out])
    got = []
    if views == ["__export__"]:
        zp = os.path.join(out, "station.zip")
        if os.path.exists(zp):
            dst = os.path.join(ROOT, "app", "station")
            shutil.rmtree(dst, ignore_errors=True)
            shutil.unpack_archive(zp, dst)
            got = ["station.zip"]
            uv = os.path.join(WORK, "apk_uv")
            shutil.rmtree(uv, ignore_errors=True)
            shutil.copytree(os.path.join(dst, "uv"), uv)
        raw = os.path.join(WORK, "lm_raw")
        os.makedirs(raw, exist_ok=True)
        lid = layout_id(os.path.join(WORK, "apk_uv"))
        if lid:
            open(os.path.join(raw, "LAYOUT"), "w").write(lid)
        for f in os.listdir(out):
            if f.endswith(".npz") or f == "progress.json":
                shutil.move(os.path.join(out, f), os.path.join(raw, f))
        if os.path.exists(os.path.join(raw, "progress.json")):
            pr = json.load(open(os.path.join(raw, "progress.json")))
            for b in pr["batches"]:
                print("  batch", b["seconds"], "s", b["msamples_per_s"], "Msamples/s", b["chunks"])
            print("chunks left for the next run:", len(pr["left"]))
        views = got[:1] or views
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
    ap.add_argument("--texel", type=float, default=0.03)
    ap.add_argument("--max-lm", type=int, default=2048)
    ap.add_argument("--max-tex", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=900, help="GPU seconds kept in reserve (export runs)")
    ap.add_argument("--tpu", action="store_true",
                    help="export runs: use a TPU machine's CPUs (separate weekly quota), baking from the last chunk")
    ap.add_argument("--no-cull", action="store_true", help="export runs: keep faces that can never be seen")
    a = ap.parse_args()
    if a.cmd == "bundle":
        bundle()
    else:
        run(a.views, a)


if __name__ == "__main__":
    main()
