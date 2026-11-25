"""Export the Umeda model for the walkthrough app (Godot, Android).

  python3 export_station.py OUT_DIR --layout-only [--check-void]     # make + check the layouts (no bake)
  python3 export_station.py OUT_DIR [--samples 128] [--texel 0.03] [--max-lm 2048] [--gpu]
        [--raw-dir DIR] [--deadline EPOCH] [--uv-from OLD/uv [--skip A,B,..]]
  python3 finish_station.py OUT_DIR RAW_DIR...      # denoise the raw lightmaps into OUT/lightmaps/

Builds the same scene as build_umeda.py (without the procedural wear layers, which cannot be
exported), then:
  1. merges the ~16,000 objects per level and fixes it for walking around (the model was built
     for still renders): faces the walker sees from behind are reversed or made two-sided, and
     where the model just stops (ceiling holes, open corridor ends, exits) ceilings, end walls
     and stairwell bulkheads are added - all found by rays from eye points over every floor the
     walker can reach,
  2. cuts it into 20 m chunks and gives each a lightmap UV layout at which 99 % of its surface
     gets texels of --texel or finer along both directions (chunks that would need more than
     --max-lm are halved); everything is kept in uv/ so another run rebuilds the same chunks,
  3. writes station.glb (chunks), collision.glb (walkable surfaces, stairs as ramps),
     materials.json + textures/ (albedo / roughness / normal / emission per material) and
     app.json (spawn points, reflection probe boxes),
  4. bakes the lighting with Cycles (direct + indirect diffuse, all area lights and glowing
     surfaces included) a few chunks at a time, writing raw float16 lightmaps as it goes and
     stopping before --deadline, so a GPU session with a time limit never loses finished work.

Lightmaps hold irradiance E encoded as sqrt(E / (E + 1)) in 8-bit PNG; the app decodes it.
Coordinates in app.json are Godot's (x, z_blender, -y_blender).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "umeda"))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("UMEDA_EXPORT", "1")
os.environ.setdefault("UMEDA_VAULT_STEP", "0.15")

import bpy  # noqa: E402  (must come before bmesh)
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import build_umeda as BU  # noqa: E402

B = BU.B
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def g(p):
    """Blender (x, y, z) -> Godot (x, z, -y)."""
    return [round(float(p[0]), 4), round(float(p[2]), 4), round(-float(p[1]), 4)]


# ------------------------------------------------------------------ scene

def build(gpu):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ns = argparse.Namespace(out="/tmp/unused.exr", width=64, samples=1, noise=0.1)
    B.render_settings(ns)
    if gpu:
        BU.enable_gpu()
    M = BU.materials()
    BU.build_b1(M)
    BU.build_b2(M)
    BU.build_stairs(M)
    BU.build_gates(M)
    BU.build_exits(M)
    BU.build_details(M)
    BU.UP.build(BU, M)
    BU.build_b1_lights(M)
    log("scene built", len(bpy.data.objects), "objects")
    return M


def level_of(ob):
    names = [c.name for c in ob.users_collection]
    for n in names:
        if n.startswith("Stairs"):
            return "ST"
        if n.startswith("B1"):
            return "B1"
        if n.startswith("B2") or n.startswith("Vault"):
            return "B2"
    z = (ob.matrix_world @ ob.data.vertices[0].co).z if ob.data.vertices else 0.0
    return "B1" if z > -2.5 else "B2"


def is_collider(ob):
    names = [c.name for c in ob.users_collection]
    if any(n.endswith("_ceiling") or n.endswith("_lights") or n == "Vault" for n in names):
        return False
    # stair treads stay solid too; the ramps laid over them make them walkable
    return True


def baked_meshes():
    """Real, world-space copies of every mesh object (modifiers applied), grouped by level."""
    dg = bpy.context.evaluated_depsgraph_get()
    groups = {"B1": [], "B2": [], "ST": []}
    coll = []
    for ob in list(bpy.data.objects):
        if ob.type != "MESH" or not ob.data.polygons:
            continue
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        me.transform(ob.matrix_world)
        if "UVMap" not in me.uv_layers:
            if len(me.uv_layers):
                me.uv_layers[0].name = "UVMap"
            else:
                me.uv_layers.new(name="UVMap")
        for extra in [u.name for u in me.uv_layers if u.name != "UVMap"]:
            me.uv_layers.remove(me.uv_layers[extra])
        groups[level_of(ob)].append(me)
        if is_collider(ob):
            coll.append(me)
    return groups, coll


def join_meshes(name, meshes, link=True):
    """Join meshes into one object with numpy array concatenation (materials merged by name)."""
    mats, index = [], {}
    V, LV, LS, MI, UV, SM = [], [], [], [], [], []
    off_v = off_l = 0
    for me in meshes:
        nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
        if npoly == 0:
            continue
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        ls = np.empty(npoly, np.int32)
        me.polygons.foreach_get("loop_start", ls)
        mi = np.empty(npoly, np.int32)
        me.polygons.foreach_get("material_index", mi)
        sm = np.empty(npoly, bool)
        me.polygons.foreach_get("use_smooth", sm)
        uv = np.zeros(nl * 2, np.float32)
        if "UVMap" in me.uv_layers:
            me.uv_layers["UVMap"].data.foreach_get("uv", uv)
        remap = []
        for m in me.materials:
            key = m.name if m is not None else "__none__"
            if key not in index:
                index[key] = len(mats)
                mats.append(m)
            remap.append(index[key])
        remap = np.array(remap or [0], np.int32)
        V.append(co)
        LV.append(lv + off_v)
        LS.append(ls + off_l)
        MI.append(remap[np.clip(mi, 0, len(remap) - 1)])
        UV.append(uv)
        SM.append(sm)
        off_v += nv
        off_l += nl
    out = bpy.data.meshes.new(name)
    npoly = sum(len(x) for x in LS)
    out.vertices.add(off_v)
    out.vertices.foreach_set("co", np.concatenate(V))
    out.loops.add(off_l)
    out.loops.foreach_set("vertex_index", np.concatenate(LV))
    out.polygons.add(npoly)
    out.polygons.foreach_set("loop_start", np.concatenate(LS))
    out.polygons.foreach_set("material_index", np.concatenate(MI))
    out.polygons.foreach_set("use_smooth", np.concatenate(SM))
    uvl = out.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.concatenate(UV))
    out.update(calc_edges=True)
    for m in mats:
        out.materials.append(m)
    ob = bpy.data.objects.new(name, out)
    if link:
        bpy.context.scene.collection.objects.link(ob)
    return ob


def ramps_mesh():
    verts, faces = [], []
    for (y0, y1, path) in BU.RAMPS:
        for (xa, za), (xb, zb) in zip(path[:-1], path[1:]):
            k = len(verts)
            verts += [(xa, y0, za + 0.01), (xb, y0, zb + 0.01), (xb, y1, zb + 0.01), (xa, y1, za + 0.01)]
            # face up whichever way the stair runs (collision is one-sided)
            faces.append((k, k + 1, k + 2, k + 3) if (xb - xa) * (y1 - y0) > 0 else (k + 3, k + 2, k + 1, k))
    me = bpy.data.meshes.new("ramps")
    me.from_pydata(verts, [], faces)
    me.update()
    return me


# ------------------------------------------------------------------ chunks

def chunk_key(level, x, y):
    cx = int(math.floor((x + 140.0) / 20.0))
    cy = int(math.floor((y + 60.0) / 30.0)) if level == "B1" else 0
    return f"{level}_{cx:02d}_{cy}"


def split_into_chunks(ob, level):
    """Cut the level mesh on the chunk planes, then one object per chunk."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for xs in np.arange(-140.0, 160.0, 20.0):
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(xs, 0, 0), plane_no=(1, 0, 0))
    if level == "B1":
        for ys in np.arange(-60.0, 70.0, 30.0):
            geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(0, ys, 0), plane_no=(0, 1, 0))
    bm.to_mesh(me)
    bm.free()
    n = len(me.polygons)
    cen = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("center", cen)
    cen = cen.reshape(n, 3)
    keys = np.array([chunk_key(level, c[0], c[1]) for c in cen])
    out = []
    for k in sorted(set(keys)):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        drop = [bm.faces[i] for i in np.nonzero(keys != k)[0]]
        bmesh.ops.delete(bm, geom=drop, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        cm = bpy.data.meshes.new(k)
        bm.to_mesh(cm)
        bm.free()
        for m in me.materials:
            cm.materials.append(m)
        co = bpy.data.objects.new(k, cm)
        bpy.context.scene.collection.objects.link(co)
        out.append(co)
    bpy.data.objects.remove(ob)
    return out


def surface_area(me):
    a = np.empty(len(me.polygons), np.float32)
    me.polygons.foreach_get("area", a)
    return float(a.sum())


LM_ANGLE = 45.0          # smart-project charts: faces within this angle of the chart's direction
LM_GUTTER_PX = 4         # empty texels between charts (the bake extends edges into them)
LM_QUANTILE = 0.99       # this share of the surface (by area) gets the texel size or finer


def lightmap_uv(ob, size):
    """Charts by smart projection (flat projections, so a texel is square on the surface) packed
    into one square with LM_GUTTER_PX texels between them at the given lightmap size."""
    me = ob.data
    if "Lightmap" in me.uv_layers:
        me.uv_layers.remove(me.uv_layers["Lightmap"])
    lm = me.uv_layers.new(name="Lightmap")
    me.uv_layers.active = lm
    for o in bpy.context.scene.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # correct_aspect would squash every chart (no image is attached, the default aspect is not 1)
    bpy.ops.uv.smart_project(angle_limit=math.radians(LM_ANGLE), margin_method="FRACTION",
                             island_margin=LM_GUTTER_PX / size, area_weight=0.0, correct_aspect=False,
                             scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.select_set(False)


def texel_sizes(me, size, layer="Lightmap"):
    """Per face: area and the texel size in metres along the face's coarsest direction
    (1 / smallest singular value of the surface -> lightmap pixel mapping)."""
    n = len(me.polygons)
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[layer].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2).astype(np.float64) * size
    ls = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    ar = np.empty(n, np.float32)
    me.polygons.foreach_get("area", ar)
    smin = np.full(n, np.inf)
    for k in range(3, int(lt.max()) + 1 if n else 3):
        m = np.nonzero(lt == k)[0]
        if not len(m):
            continue
        L = ls[m][:, None] + np.arange(k)
        P = co[lv[L]]
        U = uv[L]
        for j in range(1, k - 1):                         # fan triangles
            e1 = P[:, j] - P[:, 0]
            e2 = P[:, j + 1] - P[:, 0]
            nz = np.cross(e1, e2)
            l1 = np.linalg.norm(e1, axis=1)
            ok = (np.linalg.norm(nz, axis=1) > 1e-9) & (l1 > 1e-9)
            bx = e1 / np.maximum(l1, 1e-12)[:, None]
            by = np.cross(nz, bx)
            by /= np.maximum(np.linalg.norm(by, axis=1), 1e-12)[:, None]
            q = np.stack([np.stack([(e1 * bx).sum(1), (e1 * by).sum(1)], 1),
                          np.stack([(e2 * bx).sum(1), (e2 * by).sum(1)], 1)], 1)
            f = np.stack([U[:, j] - U[:, 0], U[:, j + 1] - U[:, 0]], 1)
            det = q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]
            ok &= np.abs(det) > 1e-12
            if not ok.any():
                continue
            qi = np.stack([np.stack([q[ok, 1, 1], -q[ok, 0, 1]], 1),
                           np.stack([-q[ok, 1, 0], q[ok, 0, 0]], 1)], 1) / det[ok][:, None, None]
            s = np.linalg.svd(np.einsum("nij,njk->nik", qi, f[ok]), compute_uv=False)
            idx = m[ok]
            smin[idx] = np.minimum(smin[idx], s[:, 1])
    tex = np.where(np.isfinite(smin) & (smin > 0), 1.0 / np.where(smin > 0, smin, 1.0), np.inf)
    return ar, tex


def texel_quantile(me, size, q=LM_QUANTILE):
    ar, tex = texel_sizes(me, size)
    m = ar > 1e-6
    if not m.any():
        return 0.0
    o = np.argsort(tex[m])
    c = np.cumsum(ar[m][o]) / ar[m].sum()
    return float(tex[m][o][min(np.searchsorted(c, q), len(o) - 1)])


def bisect_chunk(ob, axis, value):
    """Cut a chunk in two on the plane x (axis 0) or y (axis 1) = value: <name>a below, <name>b above."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    no = (1, 0, 0) if axis == 0 else (0, 1, 0)
    co = (value, 0, 0) if axis == 0 else (0, value, 0)
    bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), dist=1e-5,
                           plane_co=co, plane_no=no)
    bm.to_mesh(me)
    bm.free()
    n = len(me.polygons)
    cen = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("center", cen)
    side = cen.reshape(n, 3)[:, axis] >= value
    out = []
    for tag, keep in (("a", ~side), ("b", side)):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.nonzero(~keep)[0]], context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        cm = bpy.data.meshes.new(ob.name + tag)
        bm.to_mesh(cm)
        bm.free()
        for m in me.materials:
            cm.materials.append(m)
        o = bpy.data.objects.new(ob.name + tag, cm)
        bpy.context.scene.collection.objects.link(o)
        out.append(o)
    bpy.data.objects.remove(ob)
    return out


def apply_splits(ob, splits):
    if ob.name not in splits:
        return [ob]
    a, b = bisect_chunk(ob, *splits[ob.name])
    return apply_splits(a, splits) + apply_splits(b, splits)


def layout_chunk(ob, texel, max_lm, splits, depth=0):
    """Lightmap UVs and the smallest size (multiple of 128) at which LM_QUANTILE of the chunk's
    surface gets texels of `texel` metres or finer along both directions; a chunk that would
    need more than max_lm is cut in half (recorded in `splits`) and each half laid out alone."""
    size, best, cur = max_lm, None, None
    for _ in range(6):
        lightmap_uv(ob, size)
        cur = size
        need = max(128, int(math.ceil(size * texel_quantile(ob.data, size) / texel / 128.0)) * 128)
        if need <= size:
            best = size
            if size - need < 128:
                break
            size = need
        elif size >= max_lm or (best is not None and need >= best):
            break
        else:
            size = min(max_lm, need)
    if best is None:
        if depth >= 8:
            raise RuntimeError(f"{ob.name}: cannot reach {texel} m texels")
        n = len(ob.data.polygons)
        cen = np.empty(n * 3, np.float32)
        ob.data.polygons.foreach_get("center", cen)
        cen = cen.reshape(n, 3)
        lo, hi = cen.min(0), cen.max(0)
        axis = 0 if hi[0] - lo[0] >= hi[1] - lo[1] else 1
        value = round(float(lo[axis] + hi[axis]) / 2.0, 2)
        splits[ob.name] = [axis, value]
        out = []
        for part in bisect_chunk(ob, axis, value):
            out += layout_chunk(part, texel, max_lm, splits, depth + 1)
        return out
    if cur != best:
        lightmap_uv(ob, best)
    return [(ob, best)]


# ------------------------------------------------------------------ face orientation

def _bvh(objs):
    from mathutils.bvhtree import BVHTree
    verts, polys, off = [], [], 0
    for ob in objs:
        me = ob.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        verts += [tuple(v) for v in co.reshape(-1, 3).tolist()]
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        ls = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", lt)
        lv = (lv + off).tolist()
        polys += [lv[a:a + b] for a, b in zip(ls.tolist(), lt.tolist())]
        off += len(me.vertices)
    return BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)


def walk_grid(col, grid=0.5):
    """Every floor the walker can reach from the spawn points: floors are the upward faces of the
    collision mesh with 1.8 m headroom on a `grid` m lattice, neighbours are joined when the
    height step is under 0.4 m and nothing blocks the way at knee or chest height."""
    from mathutils import Vector
    ct = _bvh([col])
    co = np.array([v.co[:] for v in col.data.vertices])
    lo, hi = co.min(0), co.max(0)
    nx, ny = int((hi[0] - lo[0]) / grid) + 1, int((hi[1] - lo[1]) / grid) + 1
    down = Vector((0, 0, -1))
    floors = {}
    for ix in range(nx):
        for iy in range(ny):
            x, y = lo[0] + ix * grid, lo[1] + iy * grid
            z, zs, allz = hi[2] + 1.0, [], []
            while True:
                h = ct.ray_cast(Vector((x, y, z)), down, 100.0)
                if h[0] is None:
                    break
                allz.append(h[0].z)
                if h[1].z > 0.5:
                    zs.append(h[0].z)
                z = h[0].z - 0.01
            ok = [zf for zf in zs if not any(zf + 0.05 < za < zf + 1.8 for za in allz)]
            if ok:
                floors[(ix, iy)] = ok

    def blocked(x, y, z, dx, dy):
        d = Vector((dx, dy, 0)).normalized()
        return any(ct.ray_cast(Vector((x, y, z + hz)), d, math.hypot(dx, dy))[0] is not None for hz in (0.45, 1.2))

    seeds = []
    for sp in app_meta()["spawns"]:
        p = sp["pos"]
        b = (p[0], -p[2], p[1])
        ix, iy = int(round((b[0] - lo[0]) / grid)), int(round((b[1] - lo[1]) / grid))
        best = min(((abs(zf - b[2]) + 0.1 * (abs(dx) + abs(dy)), (ix + dx, iy + dy, k))
                    for dx in range(-2, 3) for dy in range(-2, 3)
                    for k, zf in enumerate(floors.get((ix + dx, iy + dy), []))), default=None)
        if best is None:
            raise RuntimeError(f"spawn {sp['name']} is not over a floor")
        seeds.append(best[1])
    seen, stack = set(seeds), list(seeds)
    while stack:
        ix, iy, k = stack.pop()
        z = floors[(ix, iy)][k]
        x, y = lo[0] + ix * grid, lo[1] + iy * grid
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for k2, z2 in enumerate(floors.get((ix + dx, iy + dy), [])):
                key = (ix + dx, iy + dy, k2)
                if key in seen or abs(z2 - z) > 0.4 or blocked(x, y, max(z, z2), dx * grid, dy * grid):
                    continue
                seen.add(key)
                stack.append(key)
    log(f"walkable floor {len(seen) * grid * grid:.0f} m2")
    return {"lo": [float(lo[0]), float(lo[1])], "grid": grid,
            "nodes": [(ix, iy, float(floors[(ix, iy)][k])) for ix, iy, k in sorted(seen)]}


def walk_eyes(wg, every=2, eye=1.6):
    """Eye points 1.6 m over the reachable floor, every `every` grid steps."""
    lo, g = wg["lo"], wg["grid"]
    return [(lo[0] + ix * g, lo[1] + iy * g, z + eye) for ix, iy, z in wg["nodes"]
            if ix % every == 0 and iy % every == 0]


def orient_faces(objs, wg, rays=512):
    """The app draws front faces only. From eye points over the walkable floors, rays in all
    directions count for every face how often it is seen from the front and from behind.
    Faces seen (almost) only from behind are reversed; faces seen from both sides get a reversed
    twin 2 mm behind them (each side its own lightmap texels). Returns face indices per object."""
    from mathutils import Vector
    eyes = walk_eyes(wg)
    tree = _bvh(objs)
    nf = [len(ob.data.polygons) for ob in objs]
    front = np.zeros(sum(nf), np.int64)
    back = np.zeros(sum(nf), np.int64)
    k = np.arange(rays) + 0.5
    zz = 1 - 2 * k / rays
    phi = k * math.pi * (3 - math.sqrt(5))
    rr = np.sqrt(1 - zz * zz)
    base = np.stack([rr * np.cos(phi), rr * np.sin(phi), zz], 1)
    rng = np.random.default_rng(1)
    for e in eyes:
        a = rng.uniform(0, 2 * math.pi)
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        o = Vector(e)
        for d in (base @ rot.T).tolist():
            dv = Vector(d)
            h = tree.ray_cast(o, dv, 300.0)
            if h[0] is not None:
                if h[1].dot(dv) < 0:
                    front[h[2]] += 1
                else:
                    back[h[2]] += 1
    out, off = {}, 0
    for ob, n in zip(objs, nf):
        f, b = front[off:off + n], back[off:off + n]
        twin = (b >= 3) & (f >= 3) & (np.minimum(f, b) >= 0.1 * np.maximum(f, b))
        flip = (b >= 3) & (b > f) & ~twin
        out[ob.name] = {"flip": np.nonzero(flip)[0].tolist(), "twin": np.nonzero(twin)[0].tolist()}
        ar = np.empty(n, np.float32)
        ob.data.polygons.foreach_get("area", ar)
        log(f"{ob.name}: reversed {ar[flip].sum():.0f} m2 ({flip.sum()} faces), "
            f"two-sided {ar[twin].sum():.0f} m2 ({twin.sum()} faces)")
        off += n
    log(f"orientation from {len(eyes)} eye points")
    return out


def apply_orientation(ob, dec, gap=0.002):
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    tw = [bm.faces[i] for i in dec.get("twin", [])]
    if dec.get("flip"):
        bmesh.ops.reverse_faces(bm, faces=[bm.faces[i] for i in dec["flip"]])
    if tw:
        normals = [f.normal.copy() for f in tw]
        ret = bmesh.ops.duplicate(bm, geom=tw)
        fmap = ret["face_map"]
        new = [fmap[f] for f in tw]
        push = {}
        for f, nrm in zip(new, normals):
            for v in f.verts:
                push.setdefault(v, []).append(nrm)
        for v, ns in push.items():
            m = sum(ns, ns[0] * 0)
            v.co -= m.normalized() * gap if m.length > 1e-9 else ns[0] * gap
        bmesh.ops.reverse_faces(bm, faces=new)
    bm.to_mesh(me)
    bm.free()
    me.update()


# ------------------------------------------------------------------ closing the model

def _front_hit(tree, o, d, dist):
    """First front face along the ray (the app does not draw back faces, so rays pass them)."""
    from mathutils import Vector
    o = Vector(o)
    for _ in range(16):
        h = tree.ray_cast(o, d, dist)
        if h[0] is None:
            return None
        if h[1].dot(d) < 0:
            return h
        dist -= (h[0] - o).length + 0.003
        if dist <= 0:
            return None
        o = h[0] + d * 0.003
    return None


def find_openings(objs, wg):
    """The model was built for still renders and simply stops where no camera looked. From each
    reachable floor cell: no ceiling above (nothing drawn within 8 m) -> a ceiling patch; a floor
    edge (neighbour not reachable) through which a level, 30 deg down, 30 or 60 deg up ray sees
    nothing -> an end wall up to the ceiling on that edge (also blocks walking off the edge)."""
    from mathutils import Vector
    tree = _bvh(objs)
    lo, g = wg["lo"], wg["grid"]
    reach = {}
    for ix, iy, z in wg["nodes"]:
        reach.setdefault((ix, iy), []).append(z)
    up = Vector((0, 0, 1))
    ceil, caps = [], []
    elev = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (0, -30, 30, 60, 75)]
    tops, closed = {}, set()
    for ix, iy, z in wg["nodes"]:
        x, y = lo[0] + ix * g, lo[1] + iy * g
        eye = (x, y, z + 1.6)
        h = _front_hit(tree, eye, up, 8.0)
        if h is None:
            top = (BU.Z_B1 + BU.H_B1) if z <= BU.Z_B1 + 0.3 else z + 3.0
            if z < BU.Z_B2 + 0.5:
                top = BU.Z_B2_CEIL
            ceil.append([ix, iy, round(top, 3)])
            closed.add((ix, iy, z))
        else:
            top = h[0].z
        tops[(ix, iy, z)] = top
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if any(abs(z2 - z) <= 0.4 for z2 in reach.get((ix + dx, iy + dy), [])):
                continue
            if any(_front_hit(tree, eye, Vector((dx * cz, dy * cz, sz)).normalized(), 80.0) is None
                   for cz, sz in elev):
                caps.append((ix, iy, z, dx, dy))
    # an end wall continues along its edge as far as the floor edge runs (no slits between end
    # walls where a single cell's rays happened to hit something)
    capset = set(caps)
    grow = list(caps)
    while grow:
        ix, iy, z, dx, dy = grow.pop()
        for sg in (1, -1):
            jx, jy = ix - dy * sg, iy + dx * sg
            for z2 in reach.get((jx, jy), []):
                key = (jx, jy, z2, dx, dy)
                if abs(z2 - z) > 0.4 or key in capset:
                    continue
                if any(abs(z3 - z2) <= 0.4 for z3 in reach.get((jx + dx, jy + dy), [])):
                    continue
                capset.add(key)
                grow.append(key)
    caps = []
    for ix, iy, z, dx, dy in sorted(capset):
        kind = "daylight" if z > BU.Z_B1 + 2.0 else ("wall_b2" if z < BU.Z_B2 + 0.5 else "wall")
        caps.append([ix, iy, dx, dy, round(z - 1.0, 3), round(tops[(ix, iy, z)] + 1.0, 3), kind])
    # floor edges with nothing under them (a slab edge over a stairwell, seen from across the
    # opening): a wall hanging down from the edge to whatever is below, drawn on both sides
    drops = []
    for ix, iy, z in wg["nodes"]:
        x, y = lo[0] + ix * g, lo[1] + iy * g
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = reach.get((ix + dx, iy + dy), [])
            if any(abs(z2 - z) <= 0.4 for z2 in nb) or not any(z2 < z - 0.4 for z2 in nb):
                continue                                   # only rims of openings to a lower floor
            edge = (x + dx * (g / 2 + 0.05), y + dy * (g / 2 + 0.05), z - 0.02)
            if _front_hit(tree, edge, Vector((dx * 0.2, dy * 0.2, -1.0)).normalized(), 400.0) is not None:
                continue
            below = tree.ray_cast(Vector(edge), Vector((0, 0, -1)), 12.0)
            depth = min((edge[2] - below[0].z) if below[0] is not None else 6.0, 6.0)
            drops.append([ix, iy, dx, dy, round(z - depth - 0.05, 3), round(z + 0.02, 3), "drop"])
    # a new ceiling higher than the existing one next to it (stairs up to the street rise past the
    # concourse ceiling): a bulkhead from the lower ceiling up, as in a real stairwell
    bulk = []
    for (ix, iy, z) in sorted(closed):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for z2 in reach.get((ix + dx, iy + dy), []):
                k2 = (ix + dx, iy + dy, z2)
                if abs(z2 - z) <= 0.4 and k2 not in closed and tops[k2] < tops[(ix, iy, z)] - 0.05:
                    bulk.append([ix, iy, dx, dy, round(tops[k2] - 0.05, 3)])
    log(f"openings: {len(ceil)} ceiling cells ({len(ceil) * g * g:.0f} m2), {len(caps)} end-wall cells, "
        f"{len(bulk)} bulkhead cells")
    log(f"   {len(drops)} hanging-wall cells under floor edges")
    return {"lo": lo, "grid": g, "ceil": ceil, "caps": caps + drops, "bulk": bulk}


def void_report(objs, wg, rays=256):
    """Share of directions in which the walker sees nothing at all, over all eye points."""
    from mathutils import Vector
    tree = _bvh(objs)
    k = np.arange(rays) + 0.5
    zz = 1 - 2 * k / rays
    phi = k * math.pi * (3 - math.sqrt(5))
    rr = np.sqrt(1 - zz * zz)
    dirs = [Vector(d) for d in np.stack([rr * np.cos(phi), rr * np.sin(phi), zz], 1).tolist()]
    eyes = walk_eyes(wg)
    void = np.array([sum(_front_hit(tree, e, d, 400.0) is None for d in dirs) / rays for e in eyes])
    log(f"empty view: mean {void.mean() * 100:.2f}%, eye points seeing any {(void > 0).mean() * 100:.1f}%, "
        f">1% {(void > 0.01).sum()}, >10% {(void > 0.1).sum()} of {len(eyes)}")
    bad = np.argsort(-void)[:15]
    for i in bad:
        if void[i] > 0:
            log("   eye", [round(v, 1) for v in eyes[i]], f"{void[i] * 100:.1f}%")
    return void


def _daylight_material():
    m = bpy.data.materials.get("daylight")
    if m is None:
        m = bpy.data.materials.new("daylight")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        b.inputs["Emission Color"].default_value = (1.0, 0.98, 0.95, 1.0)
        b.inputs["Emission Strength"].default_value = 5.0
    return m


def closure_meshes(spec):
    """Meshes for find_openings' result, by level: ceilings face down, end walls face the floor
    they close. Ceiling corners take the highest of the cells around them (no gaps over stairs)."""
    lo, g = spec["lo"], spec["grid"]
    heights = {(ix, iy): h for ix, iy, h in spec["ceil"]}
    corner = {}
    for (ix, iy), h in heights.items():
        for cx in (0, 1):
            for cy in (0, 1):
                k = (ix + cx, iy + cy)
                corner[k] = max(corner.get(k, -1e9), h)

    def level_of_z(z):
        if z < BU.Z_B2 + 0.5:
            return "B2"
        return "B1" if abs(z - (BU.Z_B1 + BU.H_B1)) < 0.05 else "ST"

    parts = {}

    def add(level, mat, quad, uvs, collide=False):
        p = parts.setdefault((level, mat, collide), ([], [], []))
        k = len(p[0])
        p[0].extend(quad)
        p[1].append((k, k + 1, k + 2, k + 3))
        p[2].extend(uvs)

    for (ix, iy), h in heights.items():
        xs = (lo[0] + (ix - 0.5) * g, lo[0] + (ix + 0.5) * g)
        ys = (lo[1] + (iy - 0.5) * g, lo[1] + (iy + 0.5) * g)
        q = [(xs[0], ys[0], corner[(ix, iy)]), (xs[0], ys[1], corner[(ix, iy + 1)]),
             (xs[1], ys[1], corner[(ix + 1, iy + 1)]), (xs[1], ys[0], corner[(ix + 1, iy)])]   # faces down
        add(level_of_z(h), "ceil", q, [(v[0] / 1.2, v[1] / 1.2) for v in q])
    walls_spec = [list(c) for c in spec["caps"]]
    for ix, iy, dx, dy, z0 in spec.get("bulk", []):
        ex = [(ix + (dx > 0) + (dx == 0) * c, iy + (dy > 0) + (dy == 0) * c) for c in (0, 1)]   # edge corners
        z1 = max(corner.get(e, z0) for e in ex) + 0.02
        level = "B2" if z0 < BU.Z_B2_CEIL + 0.5 else "B1"
        walls_spec.append([ix, iy, dx, dy, z0, round(z1, 3), "wall_b2" if level == "B2" else "wall"])
    for ix, iy, dx, dy, z0, z1, kind in [w for w in walls_spec if w[6] == "drop"]:
        for sgn in (1, -1):                                # both faces, 2 mm apart
            walls_spec.append([ix, iy, dx * sgn, dy * sgn, z0, z1, "drop_in" if sgn > 0 else "drop_out"])
    walls_spec = [w for w in walls_spec if w[6] != "drop"]
    for ix, iy, dx, dy, z0, z1, kind in walls_spec:
        if kind == "drop_out":                             # stands on the same edge, faces the other way
            ix, iy = ix - dx, iy - dy
        cx, cy = lo[0] + (ix + 0.5 * dx) * g, lo[1] + (iy + 0.5 * dy) * g
        if kind.startswith("drop"):
            cx -= dx * 0.001
            cy -= dy * 0.001
        tx, ty = -dy, dx                                  # along the edge
        a = (cx - tx * g / 2, cy - ty * g / 2)
        b = (cx + tx * g / 2, cy + ty * g / 2)
        q = [(a[0], a[1], z0), (a[0], a[1], z1), (b[0], b[1], z1), (b[0], b[1], z0)]
        nrm = np.cross(np.subtract(q[1], q[0]), np.subtract(q[3], q[0]))
        if nrm[0] * dx + nrm[1] * dy > 0:                 # must face back towards the floor
            q = q[::-1]
        t0 = (a[0] * tx + a[1] * ty)
        uvs = [((t0 + (0 if v[:2] == a else g)) / 2.4, v[2] / 2.4) for v in q]
        if kind.startswith("drop"):
            level = "B2" if z1 < BU.Z_B2 + 1.0 else ("B1" if z1 > BU.Z_B1 - 0.5 else "ST")
        else:
            level = "B2" if kind == "wall_b2" else ("ST" if kind == "daylight" else "B1")
        mat = {"wall": "b1_wall", "wall_b2": "wall_white", "daylight": "daylight",
               "drop_in": "b1_wall", "drop_out": "b1_wall"}[kind]
        add(level, mat, q, uvs, collide=not kind.startswith("drop"))
    out, walls = {}, []
    for (level, mat, collide), (verts, faces, uvs) in parts.items():
        me = bpy.data.meshes.new(f"close_{level}_{mat}_{int(collide)}")
        me.from_pydata(verts, [], faces)
        me.update()
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.array([uvs[li] for li in [lp.vertex_index for lp in me.loops]], np.float32).ravel())
        m = _daylight_material() if mat == "daylight" else bpy.data.materials.get(mat)
        if m is None:
            raise RuntimeError(f"closure material {mat} not in the scene")
        me.materials.append(m)
        out.setdefault(level, []).append(me)
        if collide:
            walls.append(me)
    return out, walls


# ------------------------------------------------------------------ hidden faces

class EyeSpace:
    """Where the walker's eyes can be: a layer 1.35-1.85 m above every upward-facing collision
    surface (floors, stair/escalator ramps, platforms), on a 0.25 m voxel grid."""

    def __init__(self, col, vox=0.25):
        me = col.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        self.vox = vox
        self.lo = co.min(0) - 2.0
        hi = co.max(0) + 3.0
        self.shape = tuple(int(v) for v in np.ceil((hi - self.lo) / vox))
        self.grid = np.zeros(self.shape, bool)
        tris = []
        for poly in me.polygons:
            if poly.normal.z > 0.5:
                vs = list(poly.vertices)
                for k in range(1, len(vs) - 1):
                    tris.append((vs[0], vs[k], vs[k + 1]))
        step = 0.15
        for a, b, c in tris:
            A, Bv, C = co[a], co[b], co[c]
            n = int(np.ceil(max(np.linalg.norm(Bv - A), np.linalg.norm(C - A), np.linalg.norm(C - Bv)) / step)) + 1
            u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
            m = (u + v) <= 1.0
            u, v = u[m], v[m]
            pts = A + np.outer(u, Bv - A) + np.outer(v, C - A)
            for dz in np.arange(1.35, 1.86, vox / 2):
                q = pts + (0.0, 0.0, dz)
                ix = ((q - self.lo) / vox).astype(np.int64)
                self.grid[ix[:, 0], ix[:, 1], ix[:, 2]] = True
        log(f"eye space: {int(self.grid.sum())} voxels over {len(tris)} walkable triangles")

    def crosses(self, o, d, length):
        """Does the segment o -> o + d * length pass through the eye space?"""
        t = np.arange(0.05, length, self.vox * 0.5)
        if len(t) == 0:
            return False
        q = (np.asarray(o) + np.outer(t, d) - self.lo) / self.vox
        ix = q.astype(np.int64)
        ok = (ix >= 0).all(1) & (ix[:, 0] < self.shape[0]) & (ix[:, 1] < self.shape[1]) & (ix[:, 2] < self.shape[2])
        ix = ix[ok]
        return bool(self.grid[ix[:, 0], ix[:, 1], ix[:, 2]].any())


def _face_points(co, verts, area, per=0.04, cap=128):
    """The centre plus stratified points over the polygon (about one per `per` m2)."""
    P = co[verts]
    cen = P.mean(0)
    n = int(min(cap, np.ceil(area / per)))
    out = [cen]
    if n <= 1:
        return out
    tris = [(P[0], P[k], P[k + 1]) for k in range(1, len(P) - 1)]
    ta = np.array([0.5 * np.linalg.norm(np.cross(b - a, c - a)) for a, b, c in tris])
    if ta.sum() <= 0:
        return out
    cnt = np.maximum(1, np.round(n * ta / ta.sum())).astype(int)
    for (a, b, c), k in zip(tris, cnt):
        g = int(np.ceil(np.sqrt(2 * k)))
        for i in range(g):
            for j in range(g - i):
                u, v = (i + 1 / 3) / g, (j + 1 / 3) / g
                p = a + u * (b - a) + v * (c - a)
                out.append(cen + (p - cen) * 0.98)
    return out


def cull_hidden(objs, col, rays=16, dry=False):
    """Delete faces the walker can never see.

    The app draws front faces only, so a face can be seen only from the space in front of it.
    From points spread over each face, rays go out over its front hemisphere. A face is kept
    if any ray lands on the front of another surface (open space between the two) or passes
    through the eye space (EyeSpace) before something blocks it. Faces that only look into solid
    ground, closed cavities or empty space outside the station are removed.
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    eyes = EyeSpace(col)
    verts, polys, off = [], [], 0
    for ob in objs:
        me = ob.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        verts += [tuple(v) for v in co.reshape(-1, 3).tolist()]
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        ls = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", lt)
        lv = (lv + off).tolist()
        polys += [lv[a:a + b] for a, b in zip(ls.tolist(), lt.tolist())]
        off += len(me.vertices)
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)
    # fixed set of hemisphere directions (Fibonacci spiral, no grazing rays)
    k = np.arange(rays) + 0.5
    z = 0.08 + 0.92 * (1.0 - k / rays)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)

    def seen_from(o, dirs):
        for d in dirs:
            dv = Vector(d)
            org = Vector(o)
            hit = tree.ray_cast(org, dv, 400.0)
            if hit[0] is None:
                if eyes.crosses(o, d, 150.0):
                    return True
                continue
            if hit[1].dot(dv) < 0.0:          # the front of another surface: open space in between
                return True
            if eyes.crosses(o, d, hit[3]):     # eye space before the blocking surface
                return True
        return False

    total = removed = 0.0
    lost = {}                 # chunk -> removed area (by face centre)
    for ob in objs:
        me = ob.data
        level = ob.name[2:]
        n = len(me.polygons)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        cen = np.empty(n * 3, np.float32)
        me.polygons.foreach_get("center", cen)
        nor = np.empty(n * 3, np.float32)
        me.polygons.foreach_get("normal", nor)
        area = np.empty(n, np.float32)
        me.polygons.foreach_get("area", area)
        cen, nor = cen.reshape(n, 3), nor.reshape(n, 3)
        hidden = np.zeros(n, bool)
        for f, poly in enumerate(me.polygons):
            nz = nor[f].astype(np.float64)
            if not nz.any():
                continue
            t = np.cross(nz, (1.0, 0.0, 0.0) if abs(nz[0]) < 0.9 else (0.0, 1.0, 0.0))
            t /= np.linalg.norm(t)
            b = np.cross(nz, t)
            dirs = (local[:, :1] * t + local[:, 1:2] * b + local[:, 2:] * nz).tolist()
            seen = False
            for p in _face_points(co, list(poly.vertices), float(area[f])):
                if seen_from((p + nz * 0.002).tolist(), dirs):
                    seen = True
                    break
            hidden[f] = not seen
        if dry:
            ob["hidden"] = hidden.tolist()
            continue
        total += float(area.sum())
        removed += float(area[hidden].sum())
        for f in np.nonzero(hidden)[0]:
            k = chunk_key(level, cen[f][0], cen[f][1])
            lost[k] = lost.get(k, 0.0) + float(area[f])
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.nonzero(hidden)[0]], context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        bm.to_mesh(me)
        bm.free()
        log(f"{ob.name}: {int(hidden.sum())} of {n} faces never visible, removed")
    log(f"hidden area removed {removed:.0f} of {total:.0f} m2 ({100 * removed / max(total, 1):.1f}%)")
    return lost


# ------------------------------------------------------------------ lightmap UVs kept between runs

def loop_keys(me):
    """A 64-bit key per face corner that does not depend on element order (Blender's mesh cuts
    do not keep the same order from run to run): the corner's vertex position, the next and
    previous corner of its face (so a reversed face differs) and the material, hashed."""
    nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    q = np.round(co.reshape(-1, 3).astype(np.float64) * 1e4).astype(np.int64)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    ls = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    mi = np.empty(npoly, np.int32)
    me.polygons.foreach_get("material_index", mi)
    poly = np.repeat(np.arange(npoly), lt)
    start, total = ls[poly], lt[poly]
    k = np.arange(nl) - start
    nxt = start + (k + 1) % total
    prv = start + (k - 1) % total
    rows = np.concatenate([q[lv], q[lv[nxt]], q[lv[prv]], mi[poly][:, None].astype(np.int64)], 1).view(np.uint64)
    h = np.full(nl, 0xcbf29ce484222325, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(rows.shape[1]):                  # FNV-1a style mix, one column at a time
            h = (h ^ rows[:, j]) * np.uint64(0x100000001b3)
            h ^= h >> np.uint64(29)
    return h


def geom_signature(me):
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return [len(me.vertices), len(me.loops), len(me.polygons), round(float(np.abs(co).sum()), 1),
            zlib.crc32(np.sort(loop_keys(me)).tobytes())]


def save_uv(ob, d):
    uv = np.empty(len(ob.data.loops) * 2, np.float32)
    ob.data.uv_layers["Lightmap"].data.foreach_get("uv", uv)
    np.savez(os.path.join(d, ob.name + ".npz"), uv=uv.reshape(-1, 2), key=loop_keys(ob.data))
    return geom_signature(ob.data)


def load_uv(ob, d, sig):
    """Reuse the lightmap layout of an earlier run, so its baked lightmaps stay valid. Corners are
    matched by loop_keys, not by index."""
    me = ob.data
    if geom_signature(me) != sig:
        raise RuntimeError(f"{ob.name}: geometry differs from the earlier run, its lightmaps would not fit")
    z = np.load(os.path.join(d, ob.name + ".npz"))
    order = np.argsort(z["key"], kind="stable")
    sk = z["key"][order]
    cur = loop_keys(me)
    i = np.clip(np.searchsorted(sk, cur), 0, len(sk) - 1)
    if not np.array_equal(sk[i], cur):
        raise RuntimeError(f"{ob.name}: {int((sk[i] != cur).sum())} face corners not found in the earlier layout")
    lm = me.uv_layers.new(name="Lightmap")
    lm.data.foreach_set("uv", z["uv"][order][i].ravel())
    me.uv_layers.active = lm


# ------------------------------------------------------------------ bake

BATCH_PX = 17_000_000        # about four 2048 lightmaps per bake call


def bake_settings(samples):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.render.bake.margin = 6
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    sc.render.bake.use_pass_color = False
    sc.render.bake.use_pass_direct = True
    sc.render.bake.use_pass_indirect = True


def bake_batch(chunks, sizes, raw_dir):
    """One Cycles bake call for a few chunks; each result is written as soon as it exists."""
    originals = {}
    images = {}
    for ob in chunks:
        img = bpy.data.images.new("LM_" + ob.name, sizes[ob.name], sizes[ob.name], alpha=False, float_buffer=True)
        images[ob.name] = img
        originals[ob.name] = list(ob.data.materials)
        for i, m in enumerate(ob.data.materials):
            mc = m.copy()
            nt = mc.node_tree
            node = nt.nodes.new("ShaderNodeTexImage")
            node.name = "BAKE_TARGET"
            node.image = img
            nt.nodes.active = node
            ob.data.materials[i] = mc
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for ob in chunks:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = chunks[0]
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, use_clear=True, margin=6)
    for ob in chunks:
        img = images[ob.name]
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(a)
        tmp = os.path.join(raw_dir, ob.name + ".tmp.npz")
        np.savez_compressed(tmp, lm=a.reshape(h, w, 4)[..., :3].astype(np.float16))
        os.replace(tmp, os.path.join(raw_dir, ob.name + ".npz"))
        for i, m in enumerate(originals[ob.name]):          # back to the shared materials
            t = ob.data.materials[i]
            ob.data.materials[i] = m
            bpy.data.materials.remove(t)
        bpy.data.images.remove(img)


def bake(chunks, sizes, samples, raw_dir, deadline=0.0, skip=(), reverse=False):
    """Bake every chunk not yet in raw_dir, in batches, and stop before `deadline` (epoch s).
    Raw lightmaps are float16 irradiance (rows bottom-up, as Blender stores them)."""
    os.makedirs(raw_dir, exist_ok=True)
    bake_settings(samples)
    todo = [ob for ob in chunks if ob.name not in skip
            and not os.path.exists(os.path.join(raw_dir, ob.name + ".npz"))]
    if reverse:                    # a second machine baking at the same time starts from the other end
        todo = todo[::-1]
    batches, cur, cur_px = [], [], 0
    for ob in todo:
        px = sizes[ob.name] ** 2
        if cur and cur_px + px > BATCH_PX:
            batches.append(cur)
            cur, cur_px = [], 0
        cur.append(ob)
        cur_px += px
    if cur:
        batches.append(cur)
    log(f"baking {len(todo)} chunks in {len(batches)} batches, {samples} samples"
        + (f", {deadline - time.time():.0f}s left" if deadline else ""))
    prog_path = os.path.join(raw_dir, "progress.json")
    prog = {"samples": samples, "batches": [], "left": [o.name for o in todo]}
    rate, work_done, time_done = 2.0e6, 0.0, 0.0      # samples/s; prior until the first batch is timed
    for b in batches:
        work = sum(sizes[o.name] ** 2 for o in b) * samples
        if deadline and time.time() + work / rate + 30 > deadline:
            log(f"time budget reached, {len(prog['left'])} chunks left for the next run")
            break
        t = time.time()
        bake_batch(b, sizes, raw_dir)
        dt = time.time() - t
        work_done += work
        time_done += dt
        rate = work_done / time_done
        for o in b:
            prog["left"].remove(o.name)
        prog["batches"].append({"chunks": [o.name for o in b], "seconds": round(dt, 1),
                                "msamples_per_s": round(work / dt / 1e6, 2)})
        json.dump(prog, open(prog_path, "w"), indent=1)
        log(f"baked {[o.name for o in b]} in {dt:.0f}s ({work / dt / 1e6:.2f} Msamples/s), "
            f"{len(prog['left'])} left")
    json.dump(prog, open(prog_path, "w"), indent=1)
    return prog


def save_lightmap(e, path):
    """e: (h, w, 3) linear irradiance, rows bottom-up (Blender order). Writes the 8-bit PNG the app reads."""
    e = np.clip(np.asarray(e, np.float32), 0, None)
    enc = np.sqrt(e / (e + 1.0))
    Image.fromarray((enc[::-1] * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(path, optimize=True)
    return float(np.percentile(e, 99))


# ------------------------------------------------------------------ materials

def _src(inp):
    return inp.links[0].from_node if inp.is_linked else None


def _uv_scale(tex):
    n = _src(tex.inputs["Vector"])
    if n is not None and n.type == "MAPPING":
        s = n.inputs["Scale"].default_value
        return [float(s[0]), float(s[1])]
    return [1.0, 1.0]


def _save_image(img, path, kind, max_size):
    """kind: srgb (linear float -> sRGB 8 bit), linear (as is), file (copy the source PNG)."""
    if kind == "file" and img.filepath:
        src = bpy.path.abspath(img.filepath)
        im = Image.open(src).convert("RGB")
    else:
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)[::-1, :, :3]
        if kind == "srgb":       # in-memory textures hold linear values
            a = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    if max(im.size) > max_size:
        s = max_size / max(im.size)
        im = im.resize((max(4, int(im.size[0] * s) // 4 * 4), max(4, int(im.size[1] * s) // 4 * 4)), Image.LANCZOS)
    im.save(path, optimize=True)


def describe_materials(used, out_dir, max_tex):
    tdir = os.path.join(out_dir, "textures")
    os.makedirs(tdir, exist_ok=True)
    saved = {}

    def tex(img, kind):
        key = (img.name, kind)
        if key not in saved:
            fn = "".join(c if c.isalnum() else "_" for c in img.name) + ("_n" if kind == "normal" else "") + ".png"
            k2 = "linear" if kind in ("normal", "linear") else kind
            if img.filepath and kind == "srgb":
                k2 = "file"
            _save_image(img, os.path.join(tdir, fn), k2, max_tex if img.size[0] < 3000 else 4096)
            saved[key] = "textures/" + fn
        return saved[key]

    table = {}
    for m in used:
        b = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
        d = {}
        if b is None:
            table[m.name] = {"color": [0.5, 0.5, 0.5], "roughness": 0.5}
            continue
        bc = b.inputs["Base Color"]
        n = _src(bc)
        if n is not None and n.type == "TEX_IMAGE" and n.image is not None:
            d["albedo"] = tex(n.image, "srgb")
            d["uv_scale"] = _uv_scale(n)
        else:
            d["color"] = [float(v) for v in bc.default_value[:3]]
        r = _src(b.inputs["Roughness"])
        if r is not None and r.type in ("SEPARATE_COLOR", "SEPRGB"):
            t = _src(r.inputs[0])
            if t is not None and t.type == "TEX_IMAGE":
                d["rough"] = tex(t.image, "linear")
        d["roughness"] = float(b.inputs["Roughness"].default_value)
        nm = _src(b.inputs["Normal"])
        if nm is not None and nm.type == "NORMAL_MAP":
            t = _src(nm.inputs["Color"])
            if t is not None and t.type == "TEX_IMAGE":
                d["normal"] = tex(t.image, "normal")
                d["normal_space"] = nm.space
        d["metallic"] = float(b.inputs["Metallic"].default_value)
        d["coat"] = float(b.inputs["Coat Weight"].default_value)
        es = float(b.inputs["Emission Strength"].default_value)
        if es > 0:
            ec = b.inputs["Emission Color"]
            n = _src(ec)
            if n is not None and n.type == "TEX_IMAGE" and n.image is not None:
                d["emission_tex"] = tex(n.image, "srgb")
            else:
                d["emission"] = [float(v) for v in ec.default_value[:3]]
            d["emission_strength"] = es
        table[m.name] = d
    return table


# ------------------------------------------------------------------ app metadata

def app_meta():
    P = BU.PLAN
    Z2, Z1 = BU.Z_B2, BU.Z_B1

    def fwd(yaw_deg):
        y = math.radians(yaw_deg)
        return g((-math.sin(y), math.cos(y), 0.0))

    st = P["stairs"]["H"]
    spawns = [
        {"name": "2番線 アーチ", "pos": g((88.5, -1.6, Z2)), "fwd": fwd(105)},
        {"name": "1番線", "pos": g((82.5, 12.6, Z2)), "fwd": fwd(94)},
        {"name": "J階段", "pos": g((P["stairs"]["J"]["x_top"] + 1.2, -7.6, Z1)), "fwd": fwd(90)},
        {"name": "北改札内 通路", "pos": g((st["x_top"] - 6.0, -5.2, Z1)), "fwd": fwd(-90)},
        {"name": "北改札", "pos": g((-132.6, -4.8, Z1)), "fwd": fwd(-116)},
    ]
    probes = []
    px0, px1 = P["plat_x"]
    yw, ye = P["edges_y"]["lo"], P["edges_y"]["up"]
    tw, te = P["track_wall_y"]["west"], P["track_wall_y"]["east"]
    V = BU.VAULT
    for x in np.arange(px0 + 7.5, px1, 15.0):
        for (y0, y1) in ((tw, 1.3), (1.3, te)):
            under = V is not None and V.x0 < x < V.x1 and y1 <= 1.3
            ztop = (V.zc - 0.3) if under else BU.Z_B2_CEIL
            c = (x, (y0 + y1) / 2, Z2 + 1.7)
            ext = (7.5, (y1 - y0) / 2, max(ztop - Z2, 2.0))
            probes.append({"pos": g(c), "size": [ext[0] * 2, ext[2] * 2, ext[1] * 2],
                           "origin": [0, -(1.7 - (ztop - Z2) / 2), 0]})
    import cv2
    walk = [np.array(p[0], np.float32).reshape(-1, 1, 2) for p in P["b1_walk"]]
    for x in np.arange(-135, 135, 15.0):
        for y in np.arange(-45, 45, 15.0):
            if any(cv2.pointPolygonTest(w, (float(x), float(y)), False) >= 0 for w in walk):
                probes.append({"pos": g((x, y, Z1 + 1.6)), "size": [15.0, 3.6, 15.0], "origin": [0, -0.1, 0]})
    return {"spawns": spawns, "probes": probes, "exposure": 0.55}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--texel", type=float, default=0.03, help="lightmap texel size in metres")
    ap.add_argument("--max-lm", type=int, default=2048)
    ap.add_argument("--max-tex", type=int, default=1024)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--raw-dir", help="where raw float16 lightmaps go (default OUT/lm_raw)")
    ap.add_argument("--zip", help="pack OUT (station data without the raw lightmaps) into this zip before baking")
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; no bake batch starts that cannot end before it")
    ap.add_argument("--uv-from", help="uv/ folder of an earlier run: reuse its lightmap layouts")
    ap.add_argument("--skip", default="", help="comma separated chunks that are already baked")
    ap.add_argument("--cull", action="store_true",
                    help="drop faces that can never be seen (experimental: has dropped visible faces)")
    ap.add_argument("--no-cull", action="store_true", help="(default; kept for older command lines)")
    ap.add_argument("--layout-only", action="store_true", help="write the layouts (uv/) and stop")
    ap.add_argument("--reverse", action="store_true", help="bake the chunks in reverse order")
    ap.add_argument("--no-bake", action="store_true", help="write the station data, bake nothing")
    ap.add_argument("--check-void", action="store_true", help="report how much empty space the walker can still see")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    raw_dir = os.path.abspath(a.raw_dir or os.path.join(out, "lm_raw"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "lightmaps"))
    os.makedirs(os.path.join(out, "uv"))

    build(a.gpu)
    groups, coll_meshes = baked_meshes()
    log("meshes", {k: len(v) for k, v in groups.items()}, "colliders", len(coll_meshes))
    # collision: everything walkable or blocking, stairs/escalators covered by ramps
    col = join_meshes("collision", coll_meshes + [ramps_mesh()])
    log("collision mesh", len(col.data.polygons), "faces")
    # remove the original objects (keep the lights for the bake)
    bpy.data.batch_remove([ob for ob in bpy.data.objects if ob.type in ("MESH", "CAMERA") and ob is not col])
    bpy.context.view_layer.update()
    log("originals removed")
    levels = {}
    for level, meshes in groups.items():
        if meshes:
            levels[level] = join_meshes(f"L_{level}", meshes)
            log(level, "faces", len(levels[level].data.polygons))
    old = json.load(open(os.path.join(a.uv_from, "index.json"))) if a.uv_from else None
    # faces the walker sees from behind: reversed or made two-sided (kept with the layouts so a
    # continued bake rebuilds exactly the same geometry)
    wg = None if old else walk_grid(col)
    orient = old["orient"] if old else orient_faces(list(levels.values()), wg)
    for name, ob in levels.items():
        apply_orientation(ob, orient.get(ob.name, {}))
    # close the places where the model just stops (ceiling holes, open corridor ends, exits)
    close = old["close"] if old else find_openings(list(levels.values()), wg)
    cm, walls = closure_meshes(close)
    for level, mes in cm.items():
        ob = levels[level]
        nob = join_meshes(f"L_{level}_c", [ob.data] + mes)
        bpy.data.objects.remove(ob)
        nob.name = f"L_{level}"
        levels[level] = nob
    if walls:
        ncol = join_meshes("collision_c", [col.data] + walls)
        bpy.data.objects.remove(col)
        ncol.name = "collision"
        col = ncol
    if a.check_void and wg is not None:
        void_report(list(levels.values()), wg)
    if a.cull and not a.no_cull:
        cull_hidden(list(levels.values()), col)
    grid = []
    for level, ob in levels.items():
        grid += split_into_chunks(ob, level)
    bpy.data.batch_remove([me for me in bpy.data.meshes if me.users == 0])
    sizes, chunks = {}, []
    if old:
        splits = old["splits"]
        for ob in grid:
            chunks += apply_splits(ob, splits)
        for ob in chunks:
            sizes[ob.name] = old["sizes"][ob.name]
            load_uv(ob, a.uv_from, old["geom"][ob.name])
    else:
        splits = {}
        for ob in grid:
            for c, s in layout_chunk(ob, a.texel, a.max_lm, splits):
                chunks.append(c)
                sizes[c.name] = s
    chunks.sort(key=lambda o: o.name)
    sigs = {ob.name: save_uv(ob, os.path.join(out, "uv")) for ob in chunks}
    q = {}
    for ob in chunks:
        ar, tex = texel_sizes(ob.data, sizes[ob.name])
        m = ar > 1e-6
        o = np.argsort(tex[m])
        cdf = np.cumsum(ar[m][o]) / max(ar[m].sum(), 1e-9)
        q[ob.name] = [round(float(tex[m][o][min(np.searchsorted(cdf, v), len(o) - 1)]) * 100, 2)
                      for v in (0.5, 0.95, 0.99)] if m.any() else [0, 0, 0]
    index = {"sizes": sizes, "geom": sigs, "splits": splits, "orient": orient, "close": close,
             "texel_cm_p50_p95_p99": q}
    index["layout_id"] = hashlib.sha1(json.dumps([sizes, sigs], sort_keys=True).encode()).hexdigest()[:12]
    json.dump(index, open(os.path.join(out, "uv", "index.json"), "w"))
    log("chunks", len(chunks), "lightmap sizes", sorted(sizes.values()),
        f"= {sum(v * v for v in sizes.values()) / 1e6:.1f} Mpx", "layout", index["layout_id"])
    worst = max(q.items(), key=lambda kv: kv[1][2])
    log(f"texel cm (coarse axis) p50/p95/p99, worst p99 {worst[0]} {worst[1]}", "reused" if old else "made")
    if a.layout_only:
        return
    # materials actually used
    used = {}
    for ob in chunks:
        idx = np.zeros(len(ob.data.polygons), np.int32)
        ob.data.polygons.foreach_get("material_index", idx)
        for i in set(idx.tolist()):
            m = ob.data.materials[i]
            if m is not None:
                used[m.name] = m
    table = describe_materials(list(used.values()), out, a.max_tex)
    json.dump(table, open(os.path.join(out, "materials.json"), "w"), ensure_ascii=False, indent=1)
    log("materials", len(table))
    # glb exports
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    common = dict(export_format="GLB", use_selection=True, export_yup=True, export_apply=False)
    for k, v in (("export_image_format", "NONE"), ("export_materials", "EXPORT"), ("export_texcoords", True),
                 ("export_normals", True), ("export_all_vertex_colors", False), ("export_unused_uvs", True)):
        if k in props:
            common[k] = v

    def export(objs, path, mats=True):
        for o in bpy.context.scene.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        kw = dict(common)
        if not mats:
            kw["export_materials"] = "NONE"
        bpy.ops.export_scene.gltf(filepath=path, **kw)

    export(chunks, os.path.join(out, "station.glb"))
    export([col], os.path.join(out, "collision.glb"), mats=False)
    meta = app_meta()
    meta["lightmaps"] = {ob.name: {"file": f"lightmaps/{ob.name}.png"} for ob in chunks}
    json.dump(meta, open(os.path.join(out, "app.json"), "w"), ensure_ascii=False, indent=1)
    log("station data written")
    if a.zip:
        shutil.make_archive(a.zip[:-4] if a.zip.endswith(".zip") else a.zip, "zip", out)
        log("packed", a.zip)
    if a.no_bake:
        return
    col.hide_render = True                     # the walk ramps must not shade anything
    skip = set(x for x in a.skip.split(",") if x)
    prog = bake(chunks, sizes, a.samples, raw_dir, a.deadline, skip, reverse=a.reverse)
    log("done", out, "left:", len(prog["left"]))


if __name__ == "__main__":
    main()
