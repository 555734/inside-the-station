"""Bake again, in the full bake scene at the full sample count and with the app's lightmap UVs,
only the faces of each chunk whose existing bake is wrong:
  sliver:<chunk>      faces whose charts lay under a sliver face's UV triangle (sliver overwrote them)
  moved:<chunk>       faces whose UVs in the layout the chunk was baked with (GitHub prep) differ
                      from the app's layout (their texels hold some other place's light)
  metal:<chunk>       faces of fully metallic materials (stainless): a metal has no diffuse part,
                      so their diffuse bake was black; baked again as dielectrics (export_station's
                      bake does so now) they get the light arriving at them
  python3 patch_bake.py OUT spec [spec ...]
Writes OUT/<chunk>.npz (patched charts only), OUT/<chunk>_sliver_mask.npy (texels to replace:
old and new places of those charts), OUT/<chunk>_targets.npy. The other faces keep their geometry
(they shade and occlude as before) but their UVs are moved off the image, so only these charts
are rendered."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["UMEDA_EXPORT"] = "1"
import export_station as E
import bpy
from PIL import Image, ImageDraw

OUT = sys.argv[1]
SPECS = [a.split(":") for a in sys.argv[2:]]
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
PREP = ROOT + "/cache/kaggle/gh_prep/layout/uv"
os.makedirs(OUT, exist_ok=True)
UVD = os.environ.get("PATCH_UV", ROOT + "/cache/kaggle/apk_uv")     # the layout the lightmaps were baked with
idx = json.load(open(UVD + "/index.json"))


def by_key(path, keys):
    z = np.load(path)
    order = np.argsort(z["key"], kind="stable")
    sk = z["key"][order]
    pos = np.clip(np.searchsorted(sk, keys), 0, len(sk) - 1)
    assert (sk[pos] == keys).mean() > 0.99, path
    return z["uv"][order][pos].astype(np.float64)


def raster(s, tris, uv):
    im = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(im)
    for t in tris:
        d.polygon([tuple(uv[li] * s) for li in t], fill=255)
    return np.asarray(im) > 0                                 # row = v * s (bottom-up, as baked)


def patch_targets(mode, name):
    ob = bpy.data.objects[name]
    me = ob.data
    s = idx["sizes"][name]
    keys = E.loop_keys(me)
    orig = by_key(f"{UVD}/{name}.npz", keys)    # app layout before collapse
    cur = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["Lightmap"].data.foreach_get("uv", cur)
    cur = cur.reshape(-1, 2).astype(np.float64)
    n = len(me.polygons)
    ls = np.empty(n, np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(n, np.int32); me.polygons.foreach_get("loop_total", lt)
    poly = np.repeat(np.arange(n), lt)
    me.calc_loop_triangles()
    tris_of = [[] for _ in range(n)]
    for t in me.loop_triangles:
        tris_of[t.polygon_index].append(tuple(t.loops))
    slivers = set(poly[np.abs(orig - cur).max(1) > 1e-6].tolist())
    if mode == "metal":
        def full_metal(m):
            b = m.node_tree.nodes.get("Principled BSDF") if (m and m.node_tree) else None
            return b is not None and b.inputs["Metallic"].default_value >= 0.999
        mats = [full_metal(sl.material) for sl in ob.material_slots]
        mi = np.empty(n, np.int32)
        me.polygons.foreach_get("material_index", mi)
        slivers = set()
        moved = set(p for p in range(n) if mi[p] < len(mats) and mats[mi[p]])
        bad = np.zeros((s, s), bool)
    elif mode == "sliver":
        moved = set()
        bad = raster(s, [t for p in slivers for t in tris_of[p]], orig)
    else:
        # faces placed elsewhere in the layout the chunk was baked with (compared as sets of UVs
        # per corner key, so corners at the same place are not taken for moved)
        prep = by_key(f"{PREP}/{name}.npz", keys)
        moved = set(poly[np.abs(prep - orig).max(1) > 1e-5].tolist()) - slivers
        # their old places: texels the bake wrote with their light (over whatever is there now)
        bad = raster(s, [t for p in moved for t in tris_of[p]], prep)
    if True:
        target = sorted(moved)
        for p in range(n):
            if p in slivers or p in moved:
                continue
            uv = cur[ls[p]:ls[p] + lt[p]] * s
            lo = np.clip(np.floor(uv.min(0)).astype(int) - 1, 0, s - 1)
            hi = np.clip(np.ceil(uv.max(0)).astype(int) + 1, 0, s - 1)
            sub = bad[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            if not sub.any():
                continue
            fm = Image.new("L", (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1), 0)
            dfm = ImageDraw.Draw(fm)
            for t in tris_of[p]:
                dfm.polygon([tuple(cur[li] * s - lo) for li in t], fill=255)
            if (np.asarray(fm) > 0)[sub].any():
                target.append(p)
        tmask = raster(s, [t for p in target for t in tris_of[p]], cur)
        mask = bad | tmask                                    # old places and new places
        np.save(f"{OUT}/{name}_target_mask.npy", tmask)
    keep = np.zeros(len(cur), bool)
    for p in target:
        keep[ls[p]:ls[p] + lt[p]] = True
    cur2 = cur.copy()
    cur2[~keep, 0] += 3.0                                     # off the image: not rasterised
    me.uv_layers["Lightmap"].data.foreach_set("uv", cur2.astype(np.float32).ravel())
    np.save(f"{OUT}/{name}_sliver_mask.npy", mask)
    np.save(f"{OUT}/{name}_targets.npy", np.array(target))
    print(f"PATCH {mode} {name}: target faces {len(target)}, texels to replace {int(mask.sum())}", flush=True)
    return ob


sys.argv = ["export_station.py", OUT + "/station", "--samples", "128", "--texel", "0.03", "--max-lm", "2048",
            "--max-tex", "1024", "--uv-from", UVD, "--raw-dir", OUT, "--no-bake"]
if os.environ.get("PATCH_REORIENT"):
    sys.argv += ["--reorient", os.environ["PATCH_REORIENT"]]
E.main()
obs = []
for mode, name in SPECS:
    if os.path.exists(f"{OUT}/{name}.npz"):
        continue
    ob = patch_targets(mode, name)
    if len(np.load(f"{OUT}/{name}_targets.npy")):
        obs.append(ob)
bpy.data.objects["collision"].hide_render = True
E.bake_settings(128)
batch, px = [], 0
for ob in obs:
    s = idx["sizes"][ob.name]
    if batch and px + s * s > 12_000_000:
        E.bake_batch(batch, {o.name: idx["sizes"][o.name] for o in batch}, OUT)
        print("PATCH baked", [o.name for o in batch], flush=True)
        batch, px = [], 0
    batch.append(ob)
    px += s * s
if batch:
    E.bake_batch(batch, {o.name: idx["sizes"][o.name] for o in batch}, OUT)
    print("PATCH baked", [o.name for o in batch], flush=True)
print("PATCH done", flush=True)
