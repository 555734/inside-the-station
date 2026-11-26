"""Builds Osaka Metro Umeda Station (Midosuji Line) from the vectorised official map and renders it.

usage: python3 build_umeda.py --view platform|concourse|overview --out ../../render/umeda_x.exr [--width 1920 --samples 300]

Dimensions taken from the map (via plan.json) and from published figures:
  platform 24.5 m wide (12.0 + 12.5 m), effective length 197/201 m, island platform 1面2線,
  2番線 side under an arch ceiling (south part) carrying a 40 m x 4 m LED screen,
  platform screen doors since 2021.
Heights that the 2D map does not give are estimates and are listed as constants below.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector, geometry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import build_scene as B  # noqa: E402  (material / mesh helpers)
import textures as T     # noqa: E402
import umeda_gfx as UG   # noqa: E402
import umeda_vault as UV  # noqa: E402
import umeda_surface as US  # noqa: E402
import umeda_props as UP  # noqa: E402
import umeda_street as UST  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
PLAN = json.load(open(os.path.join(ROOT, "cache", "umeda", "plan.json")))

# ------------------------------------------------------------- heights (estimates, metres)
Z_B1 = 0.0          # concourse floor
H_B1 = 3.0          # concourse ceiling height
Z_B2 = -5.8         # platform floor (stairs: 34 risers of 0.17 m)
Z_B2_CEIL = -2.2    # flat platform ceiling (3.6 m clear)
Z_TRACK = Z_B2 - 1.1
Z_SPRING = Z_B2 + 3.2   # arch springing line (top of the tiled track wall)
Z_CROWN = Z_B2 + 8.05  # arch crown (rises into the B1 level, hence the void on the B1 map)
X_MIN, X_MAX = -112.0, 133.0   # tunnel extent modelled beyond the platform ends

COLL = {}
RAMPS = []      # walkable ramps over stairs/escalators for the walkthrough app: (y0, y1, [(x, z), ...][, (ox, oy, angle)])
EXIT_CUTS = []  # stairwell volumes of the exits (frame matrix, lo, hi), kept clear of concourse fittings
EXPORT = bool(os.environ.get("UMEDA_EXPORT"))   # app export: no procedural shader layers (not exportable)


def coll(name):
    if name not in COLL:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
        COLL[name] = c
    return COLL[name]


def put(ob, cname):
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll(cname).objects.link(ob)
    return ob


# ------------------------------------------------------------- geometry helpers

def box_uv(me, scale=1.0):
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = (co[a] / scale, co[b] / scale)


def prism(name, poly, z0, z1, mat, cname, caps=(True, True), sides=True, uv_scale=1.0):
    rings = [[Vector((p[0], p[1], 0.0)) for p in r] for r in poly if len(r) >= 3]
    if not rings:
        return None
    try:
        tris = geometry.tessellate_polygon(rings)
    except Exception:
        tris = []
    flat = [p for r in rings for p in r]
    n = len(flat)
    verts = [(p.x, p.y, z0) for p in flat] + [(p.x, p.y, z1) for p in flat]
    faces = []
    if caps[0]:
        faces += [(t[2], t[1], t[0]) for t in tris]
    if caps[1]:
        faces += [(t[0] + n, t[1] + n, t[2] + n) for t in tris]
    if sides:
        off = 0
        for r in rings:
            m = len(r)
            for i in range(m):
                a, b = off + i, off + (i + 1) % m
                faces.append((a, b, b + n, a + n))
            off += m
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    box_uv(me, uv_scale)
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll(cname).objects.link(ob)
    return ob


def rect(name, x0, y0, x1, y1, z, mat, cname, normal_up=True, uv_scale=1.0):
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    ob = B.quad(name, pts, mat, (0, 0, 1) if normal_up else (0, 0, -1),
                {"UVMap": lambda co: (co[0] / uv_scale, co[1] / uv_scale)})
    return put(ob, cname)


def vrect(name, p0, p1, z0, z1, mat, cname, normal, uv_scale=1.0):
    pts = [(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)]
    along = 0 if abs(p1[0] - p0[0]) > abs(p1[1] - p0[1]) else 1
    ob = B.quad(name, pts, mat, normal, {"UVMap": lambda co: (co[along] / uv_scale, co[2] / uv_scale)})
    return put(ob, cname)


def bx(name, lo, hi, mat, cname, bevel=0.0):
    return put(B.box_mm(name, lo, hi, mat, bevel), cname)


def poly_centroid_area(poly):
    r = np.array(poly[0])
    x, y = r[:, 0], r[:, 1]
    a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return r.mean(0), abs(a)


# ------------------------------------------------------------- materials

def stone(name, tile, base, seed, N=2048, size=2.4, speck=0.02, rough=0.3, tint=(1.0, 0.995, 0.98)):
    path = os.path.join(T.CACHE, f"stone_{name}_{N}.npz")
    if os.path.exists(path):
        d = {k: v.astype(np.float32) for k, v in np.load(path).items()}
    else:
        tid, lu, lv, de, row, col, px = T.tile_layout(N, size, tile, tile, 0.003, seed)
        is_tile = de > 0
        t = np.clip(de / 0.0015, 0, 1)
        h = np.sqrt(1 - (1 - t) ** 2) * 0.0012 - (~is_tile) * 0.001
        g = T.fbm((N, N), [0.8, 3.0], [0.5, 0.5], seed + 1)
        h += is_tile * g * 0.00002
        nrm = T.height_to_normal(h, px)
        v = T.per_tile(tid, seed + 2, 1.0)
        rng = np.random.default_rng(seed + 3)
        dots = rng.random((N, N)).astype(np.float32)
        dark = ndimage.gaussian_filter((dots < speck).astype(np.float32), 1.0, mode="wrap")
        cloud = T.fbm((N, N), [12, 50], [0.5, 0.5], seed + 4)
        lum = base * (1 + 0.05 * cloud + 0.03 * v) * (1 - 1.5 * dark)
        alb = np.stack([lum * tint[0], lum * tint[1], lum * tint[2]], -1)
        alb = np.where(is_tile[..., None], alb, np.array([0.12, 0.12, 0.11]))
        r = np.where(is_tile, rough + 0.06 * cloud + 0.25 * (1 - t) ** 3, 0.8)
        d = {"albedo": alb.astype(np.float32), "rough": r.astype(np.float32), "normal": nrm}
        np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    return B.tiled(name, B.tex_set(name, d))



def panels(name, pw, ph, joint, color, jcolor, rough, perf=0.0, bond=0.0, N=2048, size=2.4, seed=0, var=0.015):
    """Generic panel / paver texture: rectangular units with joints, optional perforation."""
    path = os.path.join(T.CACHE, f"panel_{name}_{N}.npz")
    if os.path.exists(path):
        d = {k: v.astype(np.float32) for k, v in np.load(path).items()}
    else:
        tid, lu, lv, de, row, col, px = T.tile_layout(N, size, pw, ph, joint, seed, bond)
        is_u = de > 0
        t = np.clip(de / 0.0015, 0, 1)
        h = np.sqrt(1 - (1 - t) ** 2) * 0.001 - (~is_u) * 0.0015
        v = T.per_tile(tid, seed + 1, 1.0)
        cloud = T.fbm((N, N), [8, 40], [0.5, 0.5], seed + 2)
        base = np.array(color, np.float32)
        alb = base * (1 + var * v[..., None] + 0.02 * cloud[..., None])
        r = rough + 0.05 * cloud + 0.2 * (1 - t) ** 3
        if perf > 0:
            c = (np.arange(N, dtype=np.float32) + 0.5) * px
            U, V = np.meshgrid(c, c)
            du = np.mod(U, perf) - perf / 2
            dv = np.mod(V, perf) - perf / 2
            hole = np.clip(1 - np.sqrt(du * du + dv * dv) / (perf * 0.18), 0, 1)
            alb = alb * (1 - 0.55 * hole[..., None])
            h -= hole * 0.0004
            r = r + 0.3 * hole
        alb = np.where(is_u[..., None], alb, np.array(jcolor, np.float32))
        r = np.where(is_u, r, 0.8)
        d = {"albedo": alb.astype(np.float32), "rough": r.astype(np.float32), "normal": T.height_to_normal(h, px)}
        np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    return B.tiled(name, B.tex_set(name, d))


def glazed(name, tw, th, bond, color, gcolor, N=2048, size=2.16, seed=17):
    """Glazed ceramic tiles: cushion edges, per-tile tilt and tint so reflections break up."""
    path = os.path.join(T.CACHE, f"glazed_{name}_{N}.npz")
    if os.path.exists(path):
        d = {k: v.astype(np.float32) for k, v in np.load(path).items()}
        return B.tiled(name, B.tex_set(name, d))
    tid, lu, lv, de, row, col, px = T.tile_layout(N, size, tw, th, 0.005, seed, bond)
    is_t = de > 0
    t = np.clip(de / 0.003, 0, 1)
    h = np.sqrt(1 - (1 - t) ** 2) * 0.0025 - (~is_t) * 0.001
    sx = T.per_tile(tid, seed + 1, 0.0012)
    sy = T.per_tile(tid, seed + 2, 0.0012)
    h += is_t * (sx * (lu - tw / 2) + sy * (lv - th / 2))
    h += is_t * T.fbm((N, N), [5, 20], [0.4, 0.6], seed + 3) * 0.00003
    v = T.per_tile(tid, seed + 4, 1.0)
    cloud = T.fbm((N, N), [10, 60], [0.5, 0.5], seed + 5)
    base = np.array(color, np.float32)
    alb = base * (1 + 0.05 * v[..., None] + 0.03 * cloud[..., None])
    alb = np.where(is_t[..., None], alb, np.array(gcolor, np.float32))
    r = np.where(is_t, 0.10 + 0.04 * cloud + 0.3 * (1 - t) ** 4, 0.85)
    d = {"albedo": alb.astype(np.float32), "rough": r.astype(np.float32), "normal": T.height_to_normal(h, px)}
    np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    return B.tiled(name, B.tex_set(name, d))


def add_wear(mat, scale=0.8, alb=0.10, rough=0.10, seed=0.0):
    """Large-scale mottling (traffic wear, cleaning streaks) on a tiled material, in object space."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 6.0
    nz.inputs["Roughness"].default_value = 0.6
    if "W" in nz.inputs:
        nz.noise_dimensions = "4D"
        nz.inputs["W"].default_value = seed
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    # albedo: multiply by 1 +- alb
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 1.0 - alb
    mr.inputs["To Max"].default_value = 1.0 + alb * 0.4
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
    src = bsdf.inputs["Base Color"].links[0].from_socket
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "SCALE"
    nt.links.new(src, mul.inputs[0])
    nt.links.new(mr.outputs["Result"], mul.inputs["Scale"])
    nt.links.new(mul.outputs["Vector"], bsdf.inputs["Base Color"])
    rsrc = bsdf.inputs["Roughness"].links[0].from_socket
    mr2 = nt.nodes.new("ShaderNodeMapRange")
    mr2.inputs["To Min"].default_value = -rough
    mr2.inputs["To Max"].default_value = rough
    mr2.inputs["From Min"].default_value = 0.3
    mr2.inputs["From Max"].default_value = 0.7
    nt.links.new(nz.outputs["Fac"], mr2.inputs["Value"])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.use_clamp = True
    nt.links.new(rsrc, add.inputs[0])
    nt.links.new(mr2.outputs["Result"], add.inputs[1])
    nt.links.new(add.outputs[0], bsdf.inputs["Roughness"])
    return mat


def _uv_scale(mat, scale):
    """Insert a Mapping node after the UV Map node (box UVs are in metres)."""
    nt = mat.node_tree
    uvn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeUVMap"][0]
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, 1.0)
    for lk in list(uvn.outputs["UV"].links):
        nt.links.new(mp.outputs["Vector"], lk.to_socket)
    nt.links.new(uvn.outputs["UV"], mp.inputs["Vector"])
    return mat


def tactile_mat(name, kind, N=1024, size=0.6):
    """JIS-style tactile blocks, 30 cm: 5x5 truncated domes (warning) or 4 bars (guide line,
    running along U). Height map in metres -> normals; tops are polished by feet, the base holds dirt."""
    path = os.path.join(T.CACHE, f"tactile_{kind}_{N}.npz")
    if not os.path.exists(path):
        px = size / N
        c = (np.arange(N, dtype=np.float32) + 0.5) * px
        U, V = np.meshgrid(c, c)
        lu, lv = np.mod(U, 0.3), np.mod(V, 0.3)
        edge = np.minimum(np.minimum(lu, 0.3 - lu), np.minimum(lv, 0.3 - lv))
        if kind == "dots":
            du = np.mod(lu, 0.06) - 0.03
            dv = np.mod(lv, 0.06) - 0.03
            r = np.sqrt(du * du + dv * dv)
            prof = np.clip((0.0135 - r) / 0.0025, 0, 1)
            cap = np.clip(1 - (r / 0.011) ** 2, 0, 1)
            h = 0.004 * prof + 0.001 * cap * (r < 0.011)
        else:
            dv = np.abs(np.mod(lv, 0.075) - 0.0375)
            inbar = (lu > 0.015) & (lu < 0.285)
            endf = np.clip(np.minimum(lu - 0.015, 0.285 - lu) / 0.006, 0, 1)
            prof = np.clip((0.0135 - dv) / 0.005, 0, 1) * inbar * endf
            h = 0.005 * prof
        h = h - (edge < 0.002) * 0.001
        g = T.fbm((N, N), [1.0, 6.0], [0.5, 0.5], 7)
        h = h + g * 0.00003
        top = np.clip(h / 0.004, 0, 1)
        yel = np.array([0.80, 0.52, 0.03], np.float32)
        dirt = (1 - top) * np.clip(0.5 + 0.5 * T.fbm((N, N), [20, 60], [0.5, 0.5], 9), 0, 1)
        alb = yel * (1 - 0.25 * dirt[..., None]) * (1 + 0.04 * g[..., None])
        alb = np.where((edge < 0.002)[..., None], np.array([0.10, 0.09, 0.07], np.float32), alb)
        rough = np.clip(0.55 - 0.22 * top + 0.05 * g, 0.2, 0.9)
        d = {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": T.height_to_normal(h, px)}
        np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    d = {k: v.astype(np.float32) for k, v in np.load(path).items()}
    return _uv_scale(B.tiled(name, B.tex_set(name, d)), 1.0 / size)


def checker_mat(name="checker", N=1024, size=0.3):
    """Diamond tread steel plate for inspection covers."""
    path = os.path.join(T.CACHE, f"checker_{N}.npz")
    if not os.path.exists(path):
        px = size / N
        c = (np.arange(N, dtype=np.float32) + 0.5) * px
        U, V = np.meshgrid(c, c)
        p = 0.03
        iu, iv = np.floor(U / p), np.floor(V / p)
        fu, fv = U / p - iu - 0.5, V / p - iv - 0.5
        alt = np.mod(iu + iv, 2) == 0
        a, b = np.where(alt, fu + fv, fu - fv), np.where(alt, fu - fv, fu + fv)
        lens = np.clip(1 - (a / 0.75) ** 2 - (b / 0.16) ** 2, 0, 1)
        h = 0.0012 * np.sqrt(lens)
        g = T.fbm((N, N), [2.0, 10.0], [0.5, 0.5], 3)
        alb = np.stack([0.45 + 0.03 * g] * 3, -1) * np.array([1.0, 0.99, 0.97])
        rough = np.clip(0.38 - 0.15 * (h > 0.0003) + 0.06 * g, 0.12, 0.7)
        d = {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": T.height_to_normal(h, px)}
        np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    d = {k: v.astype(np.float32) for k, v in np.load(path).items()}
    m = _uv_scale(B.tiled(name, B.tex_set(name, d)), 1.0 / size)
    m.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 1.0
    return m


def pavers(name, pw, ph, color, gcolor, rough, seed, N=2048, size=2.4, speck=1.0):
    """Ceramic / stone pavers for close-up: granite-like speckle, per-paver tilt and tone,
    chipped edges, dirt in the joints, gloss varying paver by paver."""
    path = os.path.join(T.CACHE, f"pavers_{name}_{N}.npz")
    if not os.path.exists(path):
        tid, lu, lv, de, row, col, px = T.tile_layout(N, size, pw, ph, 0.005, seed, 0.5)
        is_u = de > 0
        t = np.clip(de / 0.0012, 0, 1)
        rng = np.random.default_rng(seed)
        # micro-relief: rounded arris, slight tilt and doming per paver
        h = np.sqrt(1 - (1 - t) ** 2) * 0.0008 - (~is_u) * 0.0015
        h += is_u * (T.per_tile(tid, seed + 1, 0.0010) * (lu - pw / 2) + T.per_tile(tid, seed + 2, 0.0010) * (lv - ph / 2))
        h += is_u * T.fbm((N, N), [2.0, 8.0], [0.5, 0.5], seed + 3) * 0.00004
        # speckle: dark and light grains of 0.5-2 mm
        n1 = rng.random((N, N)).astype(np.float32)
        dark = ndimage.gaussian_filter((n1 < 0.035).astype(np.float32), 0.7)
        light = ndimage.gaussian_filter((n1 > 0.975).astype(np.float32), 0.7)
        mid = T.fbm((N, N), [1.5, 5.0], [0.6, 0.4], seed + 4)
        v = T.per_tile(tid, seed + 5, 1.0)
        base = np.array(color, np.float32)
        lum = 1 + 0.06 * v + 0.05 * mid + speck * (0.35 * light - 0.45 * dark)
        alb = base * lum[..., None]
        # chipped edges: random notches along the arris
        chip = (de < 0.004) & (T.fbm((N, N), [1.2], [1.0], seed + 6) > 0.55)
        alb = np.where(chip[..., None], alb * 0.75, alb)
        h = np.where(chip, h - 0.0008, h)
        # dirt in the joints, a little darker in some
        jd = 0.7 + 0.3 * T.fbm((N, N), [30.0], [1.0], seed + 7)
        alb = np.where(is_u[..., None], alb, np.array(gcolor, np.float32) * jd[..., None])
        r = rough + 0.04 * T.per_tile(tid, seed + 8, 1.0) + 0.03 * mid + 0.25 * (1 - t) ** 3
        r = np.where(is_u, np.clip(r, 0.05, 0.9), 0.85)
        d = {"albedo": alb.astype(np.float32), "rough": r.astype(np.float32), "normal": T.height_to_normal(h, px)}
        np.savez(path, **{k: v_.astype(np.float16) for k, v_ in d.items()})
    d = {k: v_.astype(np.float32) for k, v_ in np.load(path).items()}
    return B.tiled(name, B.tex_set(name, d))


def materials():
    M = {}
    M["b1_floor"] = stone("b1_floor2", 0.6, 0.50, 5, rough=0.16, tint=(1.0, 1.0, 0.99))
    # platform: small beige-grey pavers in running bond (photos 2019/2022)
    M["b2_floor"] = pavers("b2s", 0.30, 0.15, (0.44, 0.43, 0.41), (0.15, 0.15, 0.14), 0.21, seed=3)
    M["wall_tile"] = B.tiled("wall_tile", B.tex_set("wall_tile", T.wall_tiles(N=2048, tile=0.20, grout=0.004, stripe_row=None, tag="_umeda")))
    M["b1_wall"] = panels("b1_wall", 1.2, 0.6, 0.006, (0.80, 0.80, 0.78), (0.25, 0.25, 0.25), 0.28, seed=5)
    M["vault"] = panels("vault_perf", 1.2, 0.6, 0.008, (0.80, 0.80, 0.78), (0.35, 0.35, 0.34), 0.45, perf=0.012, seed=7)
    M["beige"] = panels("beige_wall", 1.2, 0.9, 0.006, (0.66, 0.61, 0.52), (0.30, 0.28, 0.25), 0.3, seed=9)
    M["ceil"] = B.tiled("ceil", B.tex_set("ceil", T.ceiling_spandrel(N=2048)))
    M["panel_white"] = B.plain("panel_white", (0.80, 0.80, 0.79), 0.32)
    M["panel_grey"] = B.plain("panel_grey", (0.03, 0.03, 0.035), 0.3)
    M["column"] = B.plain("column", (0.80, 0.80, 0.79), 0.22, coat=0.3)
    M["concrete"] = B.plain("concrete", (0.10, 0.10, 0.10), 0.85)
    M["ballast"] = B.plain("ballast", (0.06, 0.06, 0.06), 0.9)
    M["rail"] = B.plain("rail", (0.55, 0.52, 0.5), 0.3, metal=1.0)
    M["third_rail_cover"] = B.plain("third_rail_cover", (0.75, 0.62, 0.15), 0.5)
    M["steel"] = B.plain("steel", (0.78, 0.78, 0.77), 0.22, metal=1.0)
    M["steel_brushed"] = B.plain("steel_brushed", (0.72, 0.72, 0.71), 0.22, metal=1.0)
    M["psd_glass"] = B.plain("psd_glass", (0.05, 0.06, 0.07), 0.05, coat=1.0)
    M["gate_body"] = B.plain("gate_body", (0.10, 0.10, 0.11), 0.3, metal=0.5)
    M["gate_side"] = B.plain("gate_side", (0.70, 0.70, 0.69), 0.17, metal=1.0)
    M["gate_top"] = B.plain("gate_top", (0.02, 0.02, 0.025), 0.12, coat=1.0)
    M["gate_blue"] = B.emissive("gate_blue", (0.1, 0.35, 1.0), 6.0)
    M["gate_green"] = B.emissive("gate_green", (0.1, 1.0, 0.25), 5.0)
    M["gate_red"] = B.emissive("gate_red", (1.0, 0.08, 0.06), 4.0)
    M["gate_panel"] = B.plain("gate_panel", (0.05, 0.12, 0.45), 0.3, coat=0.5)
    M["gate_yellow"] = B.plain("gate_yellow", (0.80, 0.55, 0.04), 0.35, coat=0.3)
    M["stair"] = stone("stair_granite", 0.6, 0.20, 31, N=1024, size=1.2, speck=0.08, rough=0.35, tint=(1.0, 1.0, 1.02))
    M["nosing"] = B.plain("nosing", (0.75, 0.6, 0.12), 0.5)
    M["glass_rail"] = B.plain("glass_rail", (0.55, 0.6, 0.6), 0.06, coat=1.0)
    M["tactile"] = tactile_mat("tactile", "dots")
    M["tactile_bar"] = tactile_mat("tactile_bar", "bars")
    M["bench"] = B.plain("bench", (0.55, 0.55, 0.56), 0.3, metal=0.8)
    M["downlight"] = B.emissive("downlight", (1.0, 0.95, 0.88), 40.0)
    M["glassbox"] = B.emissive("glassbox", (1.0, 0.96, 0.86), 6.0)
    M["cove"] = B.emissive("cove", (1.0, 0.93, 0.82), 10.0)
    M["led_line"] = B.emissive("led_line", (0.95, 0.98, 1.0), 25.0)
    gfx = UG.build_all()
    for k in ("name_board", "track1", "track2", "exit_sign", "gate_nw", "gate_ne", "gate_n", "gate_s", "exit_b1", "plat_sign"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.2)
    for k in ("ad0", "ad1", "ad2", "adp0", "adp1", "adp2"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.6)
    M["led_vision"] = B.printed("led_vision", gfx["led_vision"], rough=0.85, coat=0.0, emission=1.8)
    M["led_vision"].node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.1
    for k in ("band0", "band1", "band2"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.3)
    M["mirror"] = B.plain("mirror", (0.92, 0.92, 0.92), 0.035, metal=1.0)
    M["kiosk_black"] = B.plain("kiosk_black", (0.02, 0.02, 0.022), 0.35)
    M["kiosk_white"] = B.plain("kiosk_white", (0.80, 0.80, 0.79), 0.25, coat=0.4)
    M["louver"] = B.plain("louver", (0.05, 0.05, 0.055), 0.4, metal=0.6)
    M["track_tile"] = glazed("track_tile2", 0.36, 0.27, 0.5, (0.56, 0.53, 0.51), (0.24, 0.23, 0.22))
    M["b2_floor_band"] = pavers("b2bands", 0.30, 0.15, (0.30, 0.25, 0.23), (0.12, 0.11, 0.10), 0.22, seed=13)
    M["psd_panel"] = B.plain("psd_panel", (0.78, 0.78, 0.77), 0.3, coat=0.3)
    M["door_dark"] = B.plain("door_dark", (0.02, 0.025, 0.03), 0.08, coat=1.0)
    M["esc_clad"] = B.plain("esc_clad", (0.55, 0.54, 0.52), 0.3, metal=0.6)
    M["esc_step"] = panels("esc_step", 0.008, 0.40, 0.003, (0.30, 0.30, 0.31), (0.08, 0.08, 0.08), 0.35, N=1024, size=0.4, seed=21)
    M["handrail"] = B.plain("handrail", (0.015, 0.015, 0.017), 0.25)
    M["handrail_steel"] = B.plain("handrail_steel", (0.78, 0.78, 0.77), 0.22, metal=1.0)
    M["parapet"] = panels("parapet", 1.2, 0.6, 0.004, (0.42, 0.41, 0.39), (0.2, 0.2, 0.2), 0.3, seed=23)
    M["portal"] = B.plain("portal", (0.012, 0.012, 0.014), 0.12, coat=0.6)
    M["portal_sign"] = B.printed("portal_sign", gfx["portal_sign"], rough=0.3, coat=0.0, emission=1.4)
    M["psd_label"] = B.printed("psd_label", gfx["psd_label"], rough=0.35, coat=0.0)
    M["ceil_b2"] = panels("ceil_b2b", 1.2, 0.6, 0.006, (0.30, 0.30, 0.30), (0.08, 0.08, 0.08), 0.40, perf=0.010, seed=27)
    M["col_steel"] = B.plain("col_steel", (0.70, 0.70, 0.69), 0.20, metal=1.0)
    M["wall_white"] = glazed("wall_white", 0.60, 0.30, 0.5, (0.70, 0.70, 0.68), (0.32, 0.32, 0.31), size=2.4)
    M["clad_dark"] = B.plain("clad_dark", (0.06, 0.065, 0.07), 0.35, metal=0.3)
    M["navy"] = B.plain("navy", (0.02, 0.03, 0.12), 0.35, coat=0.3)
    for k in ("name_board_w", "name_board_e"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.3)
    for k in ("dep1", "dep2"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=2.0)
    M["clock"] = B.printed("clock", gfx["clock"], rough=0.25, coat=0.5, emission=0.6)
    M["vault_metal"] = panels("vault_metal2", 1.2, 0.5, 0.004, (0.72, 0.72, 0.71), (0.12, 0.12, 0.12), 0.20, seed=33)
    M["vault_metal"].node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.55
    M["led_ring"] = B.emissive("led_ring", (1.0, 0.98, 0.95), 18.0)
    M["b1_sign_down"] = B.printed("b1_sign_down", gfx["b1_sign_down"], rough=0.3, coat=0.0, emission=1.2)
    M["gate_fascia_n"] = B.printed("gate_fascia_n", gfx["gate_fascia_n"], rough=0.3, coat=0.0, emission=1.5)
    M["cctv"] = B.plain("cctv", (0.02, 0.02, 0.025), 0.05, coat=1.0)
    M["exit_green"] = B.printed("exit_green", gfx["exit_green"], rough=0.3, coat=0.0, emission=3.0)
    for k in ("queue0", "queue1"):
        M[k] = B.printed(k, gfx[k], rough=0.35, coat=0.0)
    M["decal_up"] = B.printed("decal_up", gfx["decal_up"], rough=0.5, coat=0.0)
    M["decal_down"] = B.printed("decal_down", gfx["decal_down"], rough=0.5, coat=0.0)
    for k, sc, a, r, sd in (("b2_floor", 0.7, 0.09, 0.10, 1.0), ("b2_floor_band", 0.7, 0.09, 0.10, 1.0),
                            ("b1_floor", 0.5, 0.07, 0.08, 2.0), ("track_tile", 1.5, 0.05, 0.05, 3.0)):
        if k in M and not EXPORT:
            add_wear(M[k], sc, a, r, sd)
    M["lamp_red"] = B.emissive("lamp_red", (1.0, 0.08, 0.05), 6.0)
    M["lamp_amber"] = B.emissive("lamp_amber", (1.0, 0.45, 0.05), 5.0)
    M["paint_red"] = B.plain("paint_red", (0.55, 0.03, 0.03), 0.3, coat=0.5)
    M["paint_yellow"] = B.plain("paint_yellow", (0.80, 0.60, 0.02), 0.35, coat=0.3)
    M["checker"] = checker_mat()
    M["screen_bezel"] = B.plain("screen_bezel", (0.01, 0.01, 0.012), 0.08, coat=1.0)
    for k in ("psd_sticker", "ic_reader", "gate_ok", "gate_ng", "ic_only", "hydrant", "aed", "estop", "asset"):
        em = {"ic_reader": 1.5, "gate_ok": 3.0, "gate_ng": 3.0}.get(k, 0.0)
        M[k] = B.printed(k, gfx[k], rough=0.35, coat=0.3 if k not in ("ic_only",) else 0.0, emission=em)
    if not EXPORT:
        US.apply_all(M)
    return M



def build_b1(M):
    c = "B1"
    for i, poly in enumerate(PLAN["b1_floor"]):
        prism(f"b1_floor_{i}", poly, Z_B1 - 0.35, Z_B1, M["b1_floor"], c, uv_scale=2.4)
    for i, poly in enumerate(PLAN["b1_walls"]):
        cen, area = poly_centroid_area(poly)
        if area < 2.5:   # free-standing columns: enamel cladding with stainless skirting
            prism(f"b1_col_{i}", poly, Z_B1 + 0.15, Z_B1 + H_B1, M["column"], c)
            prism(f"b1_colsk_{i}", poly, Z_B1, Z_B1 + 0.15, M["steel_brushed"], c)
        else:
            prism(f"b1_wall_{i}", poly, Z_B1, Z_B1 + H_B1 + 1.1, M["b1_wall"], c, uv_scale=2.4)
    # the ceiling outline was taken from the floor's, stair openings and all: the concourse ceiling
    # runs on over the stairs (only the outer ring)
    for i, poly in enumerate(PLAN["b1_ceiling"]):
        prism(f"b1_ceil_{i}", poly[:1], Z_B1 + H_B1, Z_B1 + H_B1 + 0.05, M["ceil"], "B1_ceiling", uv_scale=1.2)
    build_b1_corridor(M)
    # hanging signs over the gate lines
    names = {"北改札": "gate_n", "中北西改札": "gate_nw", "中北東改札": "gate_ne", "南改札": "gate_s"}
    unpaid_east = {"中北西改札", "南改札"}     # side of the gate line facing the unpaid area
    for name, pl in PLAN["gates"].items():
        pts = np.array(pl)
        mid = pts[len(pts) // 2 - 1:len(pts) // 2 + 1].mean(0) if len(pts) > 2 else pts.mean(0)
        if name in unpaid_east:
            hang_sign(f"gs_{name}", float(mid[0]) + 1.2, float(mid[1]), M["exit_b1"], M[names[name]], c)
        else:
            hang_sign(f"gs_{name}", float(mid[0]) - 1.2, float(mid[1]), M[names[name]], M["exit_b1"], c)
    hang_sign("hs_exit", 9.5, 2.0, M["exit_b1"], M["exit_b1"], c, along="y")


def build_b1_corridor(M):
    """Corridor inside the north gate (2023 photo): barrel vault of metal panels with transverse
    light rings, a dark recess at the springing, white tiled walls, black hanging signs."""
    c = "B1"
    cr = PLAN["b1_corridor"]
    x0, x1, yn, ys = cr["x0"], cr["x1"], cr["y_n"], cr["y_s"]
    zs = Z_B1 + H_B1 - 0.05
    prof, (ym, zcen, R, a0) = arch_profile(ys + 0.15, yn - 0.15, zs + 0.35, Z_B1 + H_B1 + 1.3, n=40)
    verts = []
    for x in (x0, x1):
        verts += [(x, y, z) for (y, z) in prof]
    m = len(prof)
    faces = [(i, m + i, m + i + 1, i + 1) for i in range(m - 1)]
    me = bpy.data.meshes.new("b1_vault")
    me.from_pydata(verts, [], faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            j_, i_ = divmod(vi, m)
            uvl.data[li].uv = ((x0, x1)[j_] / 2.4, (R * (2 * a0 * i_ / (m - 1))) / 2.4)
    me.materials.append(M["vault_metal"])
    coll("B1_ceiling").objects.link(bpy.data.objects.new("b1_vault", me))
    # dark recess between the wall tops and the vault edge
    for yy, nm in ((ys, "s"), (yn - 0.18, "n")):
        bx(f"b1_recess_{nm}", (x0, yy, zs - 0.02), (x1, yy + 0.18, zs + 0.38), M["panel_grey"], "B1_ceiling")
    # transverse light rings every 2.4 m
    k = 0
    for xr in np.arange(x0 + 0.9, x1 - 0.6, 1.8):
        pts = []
        for (yy, zz) in prof[2:-2]:
            pts.append((yy, zz))
        for q in range(len(pts) - 1):
            (ya, za), (yb, zb) = pts[q], pts[q + 1]
            ob = B.box(f"b1ring_{k}_{q}", (0, 0, 0), (0.05, math.hypot(yb - ya, zb - za) + 0.004, 0.02), M["led_ring"])
            ob.location = (xr, (ya + yb) / 2 - 0.01 * math.copysign(1, (ya + yb) / 2 - ym), (za + zb) / 2 - 0.012)
            ob.rotation_euler = (math.atan2(zb - za, yb - ya), 0, 0)
            put(ob, "B1_lights")
        L = B.area_light(f"b1ringA_{k}", (xr, ym, Z_B1 + H_B1 + 1.2), 0.1, (yn - ys) * 0.8, 140.0, (1.0, 0.98, 0.95), camera=False)
        put(L, "B1_lights")
        k += 1
    # ads along both corridor walls, tactile guide line
    for i, x in enumerate(np.arange(x0 + 5, x1 - 3, 5.5)):
        for y, right, nm in ((yn - 0.45, (1, 0, 0), "n"), (ys + 0.25, (-1, 0, 0), "s")):
            B.image_quad(f"b1ad_{nm}_{i}", (x, y, Z_B1 + 1.55), right, (0, 0, 1), 2.6, 1.45, M[f"ad{(i + (nm == 's')) % 3}"])
            bx(f"b1adf_{nm}_{i}", (x - 1.36, y + 0.002 if nm == "n" else y - 0.25, Z_B1 + 0.78), (x + 1.36, y + 0.25 if nm == "n" else y - 0.002, Z_B1 + 2.32), M["steel"], c)
    for ob in list(bpy.context.scene.collection.objects):
        if ob.name.startswith("b1ad_"):
            put(ob, c)
    # tactile guide line along the corridor, interrupted at the stair openings
    yt = (yn + ys) / 2 + 1.2
    cuts = sorted((min(PLAN["stairs"][k]["x_top"], PLAN["stairs"][k]["x_bot"]) - 0.6,
                   max(PLAN["stairs"][k]["x_top"], PLAN["stairs"][k]["x_bot"]) + 0.6)
                  for k in ("F", "G", "H")
                  if PLAN["stairs"][k]["y0"] - 0.2 < yt < PLAN["stairs"][k]["y1"] + 0.2)
    xa = x0 + 1
    for (ca, cb) in cuts + [(x1 - 1, x1)]:
        if ca > xa:
            bx(f"b1_tactile_corr_{xa:.0f}", (xa, yt - 0.15, Z_B1), (ca, yt + 0.15, Z_B1 + 0.005), M["tactile_bar"], c)
        xa = max(xa, cb)
    # black hanging signs before each stair
    for k in ("F", "G", "H"):
        st = PLAN["stairs"][k]
        xs = st["x_top"] - 0.8
        hang_sign(f"ps_{k}", xs, (yn + ys) / 2 + 0.8, M["b1_sign_down"], M["b1_sign_down"], c, z_top=Z_B1 + H_B1 + 1.2, w=4.2, h=0.62)


def hang_sign(name, x, y, mat_front, mat_back, cname, along="x", z_top=None, w=2.6, h=0.48):
    z_top = Z_B1 + H_B1 if z_top is None else z_top
    zc = Z_B1 + H_B1 - 0.55
    if along == "x":    # sign faces -x / +x
        B.image_quad(name + "_f", (x - 0.06, y, zc), (0, -1, 0), (0, 0, 1), w, h, mat_front)
        B.image_quad(name + "_b", (x + 0.06, y, zc), (0, 1, 0), (0, 0, 1), w, h, mat_back)
        bx(name + "_body", (x - 0.055, y - w / 2 - 0.03, zc - h / 2 - 0.03), (x + 0.055, y + w / 2 + 0.03, zc + h / 2 + 0.03), bpy.data.materials["panel_grey"], cname)
        rods = [(x, y - 0.9), (x, y + 0.9)]
    else:               # sign faces -y / +y
        B.image_quad(name + "_f", (x, y + 0.06, zc), (-1, 0, 0), (0, 0, 1), w, h, mat_front)
        B.image_quad(name + "_b", (x, y - 0.06, zc), (1, 0, 0), (0, 0, 1), w, h, mat_back)
        bx(name + "_body", (x - w / 2 - 0.03, y - 0.055, zc - h / 2 - 0.03), (x + w / 2 + 0.03, y + 0.055, zc + h / 2 + 0.03), bpy.data.materials["panel_grey"], cname)
        rods = [(x - 0.9, y), (x + 0.9, y)]
    for rx, ry in rods:
        bx(name + f"_rod{rx:.1f}{ry:.1f}", (rx - 0.01, ry - 0.01, zc + h / 2), (rx + 0.01, ry + 0.01, z_top), bpy.data.materials["steel"], cname)
    for ob in list(bpy.context.scene.collection.objects):
        if ob.name.startswith(name):
            put(ob, cname)


def build_b2(M):
    c = "B2"
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    yv_w, yv_e = PLAN["vault"]["y_west"], PLAN["vault"]["y_east"]
    for i, poly in enumerate(PLAN["b2_floor"]):
        prism(f"b2_floor_{i}", poly, Z_TRACK, Z_B2, M["b2_floor"], c, uv_scale=2.4)
    # the vectorised floor stops a little short of the platform edges: close the gap
    px0_, px1_ = PLAN["plat_x"]
    for yE, sgn_, nm in ((PLAN["edges_y"]["up"], -1, "e"), (PLAN["edges_y"]["lo"], 1, "w")):
        y_a, y_b = sorted((yE, yE + sgn_ * 1.6))
        rect(f"b2_edgefill_{nm}", px0_, y_a, px1_, y_b, Z_B2 - 0.0015, M["b2_floor"], c, True, 2.4)
    for i, poly in enumerate(PLAN["b2_walls"]):
        cen, area = poly_centroid_area(poly)
        in_vault = vx0 < cen[0] < vx1 and yv_w < cen[1] < yv_e
        if in_vault and area < 3.0:
            r = np.array(poly[0])
            kiosk(f"kiosk_{i}", r.min(0) + 0.2, r.max(0) - 0.2, M, i)
            continue
        r = np.array(poly[0])
        lo_, hi_ = r.min(0), r.max(0)
        d = hi_ - lo_
        cxy = (lo_ + hi_) / 2
        if max(d) <= 1.0 and min(d) >= 0.6:                 # round stainless column (photos)
            dia = min(d) - 0.22
            put(B.cylinder(f"b2_rcol_{i}", (cxy[0], cxy[1], (Z_B2 + Z_B2_CEIL) / 2), dia / 2, Z_B2_CEIL - Z_B2, M["col_steel"], segs=48), c)
            put(B.cylinder(f"b2_rcolb_{i}", (cxy[0], cxy[1], Z_B2 + 0.05), dia / 2 + 0.01, 0.10, M["steel"], segs=48), c)
            continue
        if 1.0 < max(d) <= 1.35 and min(d) >= 1.0:          # square pillar in dark grey cladding with a screen
            sq_pillar(f"b2_sq_{i}", lo_ + 0.15, hi_ - 0.15, M, i)
            continue
        if area < 1.1 and min(d) <= 0.9 and max(d) <= 1.2:   # navy signage kiosk
            navy_kiosk(f"b2_nk_{i}", lo_ + 0.12, hi_ - 0.12, M, i)
            continue
        mat = M["column"] if area < 1.5 else M["wall_white"]
        prism(f"b2_wall_{i}", poly, Z_B2, Z_B2_CEIL, mat, c, uv_scale=2.4)
    for i, poly in enumerate(PLAN["b2_ceiling"]):
        prism(f"b2_ceil_{i}", poly, Z_B2_CEIL, Z_B2_CEIL + 0.05, M["ceil_b2"], "B2_ceiling", uv_scale=2.4)

    # track beds, rails, tunnel walls
    ye, yw = PLAN["edges_y"]["up"], PLAN["edges_y"]["lo"]
    te, tw = PLAN["track_wall_y"]["east"], PLAN["track_wall_y"]["west"]
    rect("trackbed_e", X_MIN, ye, X_MAX, te, Z_TRACK, M["ballast"], c)
    rect("trackbed_w", X_MIN, tw, X_MAX, yw, Z_TRACK, M["ballast"], c)
    px0, px1 = PLAN["plat_x"]
    for y, nrm, nm in ((ye, (0, 1, 0), "e"), (yw, (0, -1, 0), "w")):
        vrect(f"plat_edge_{nm}", (px0, y), (px1, y), Z_TRACK, Z_B2, M["concrete"], c, nrm)
    vrect("tunnel_e", (X_MIN, te), (X_MAX, te), Z_TRACK, Z_B2_CEIL, M["track_tile"], c, (0, -1, 0), 2.16)
    # west wall: up to the arch springing under the vault, else to the flat ceiling
    vrect("tunnel_w_n", (X_MIN, tw), (vx0, tw), Z_TRACK, Z_B2_CEIL, M["track_tile"], c, (0, 1, 0), 2.16)
    vrect("tunnel_w_v", (vx0, tw), (vx1, tw), Z_TRACK, Z_SPRING, M["track_tile"], c, (0, 1, 0), 2.16)
    vrect("tunnel_w_s", (vx1, tw), (X_MAX, tw), Z_TRACK, Z_B2_CEIL, M["track_tile"], c, (0, 1, 0), 2.16)
    rect("ceil_track_e_n", X_MIN, ye, px0, te, Z_B2_CEIL, M["ceil_b2"], "B2_ceiling", False, 1.2)
    rect("ceil_track_w_n", X_MIN, tw, px0, yw, Z_B2_CEIL, M["ceil_b2"], "B2_ceiling", False, 1.2)
    rect("ceil_track_e_s", px1, ye, X_MAX, te, Z_B2_CEIL, M["ceil_b2"], "B2_ceiling", False, 1.2)
    rect("ceil_track_w_s", px1, tw, X_MAX, yw, Z_B2_CEIL, M["ceil_b2"], "B2_ceiling", False, 1.2)
    for yc, side, nm in ((ye + 1.85, 1, "e"), (yw - 1.75, -1, "w")):
        for g in (-0.7175, 0.7175):
            bx(f"rail_{nm}{g}", (X_MIN, yc + g - 0.035, Z_TRACK + 0.18), (X_MAX, yc + g + 0.035, Z_TRACK + 0.33), M["rail"], c)
        bx(f"slab_{nm}", (X_MIN, yc - 1.1, Z_TRACK), (X_MAX, yc + 1.1, Z_TRACK + 0.18), M["concrete"], c)
        y3 = yc + side * 1.45
        bx(f"third_{nm}", (X_MIN, y3 - 0.05, Z_TRACK + 0.2), (X_MAX, y3 + 0.05, Z_TRACK + 0.42), M["third_rail_cover"], c)
    # tunnel portals beyond the platform: dark ends
    for x, nm in ((X_MIN, "n"), (X_MAX, "s")):
        vrect(f"tunnel_end_e_{nm}", (x, ye), (x, te), Z_TRACK, Z_B2_CEIL, M["concrete"], c, (1 if x < 0 else -1, 0, 0))
        vrect(f"tunnel_end_w_{nm}", (x, tw), (x, yw), Z_TRACK, Z_B2_CEIL, M["concrete"], c, (1 if x < 0 else -1, 0, 0))
    build_vault(M)
    floor_band(M)
    guide_lines(M)
    build_psd(M)
    build_tactile(M)
    build_b2_lights(M)
    build_b2_signs(M)


def arch_profile(yw, ye, zs, zc, n=48):
    span = ye - yw
    rise = zc - zs
    R = (span * span / 4 + rise * rise) / (2 * rise)
    ym = (yw + ye) / 2
    zcen = zc - R
    a0 = math.asin((span / 2) / R)
    pts = []
    for i in range(n + 1):
        a = -a0 + 2 * a0 * i / n
        pts.append((ym + R * math.sin(a), zcen + R * math.cos(a)))
    return pts, (ym, zcen, R, a0)



VAULT = None


def vault_z(y):
    """Height of the barrel at plan position y (under the vault)."""
    V = VAULT
    s = V.s_of_y(np.array([y]))
    return float(V.at_s(s)[1][0])


def sq_pillar(name, lo, hi, M, idx):
    """Square column clad in dark grey panels with a portrait screen on two faces (wp_plat1)."""
    c = "B2"
    (x0, y0), (x1, y1) = lo, hi
    bx(name, (x0, y0, Z_B2), (x1, y1, Z_B2_CEIL), M["clad_dark"], c, 0.01)
    bx(name + "_vent", (x0 - 0.005, y0 + 0.1, Z_B2 + 2.85), (x1 + 0.005, y1 - 0.1, Z_B2 + 3.1), M["louver"], c)
    for j, (fx, right) in enumerate(((x0 - 0.006, (0, -1, 0)), (x1 + 0.006, (0, 1, 0)))):
        put(B.image_quad(f"{name}_scr{j}", (fx, (y0 + y1) / 2, Z_B2 + 1.65), right, (0, 0, 1), (y1 - y0) - 0.16, ((y1 - y0) - 0.16) * 1.78, M[f"adp{(idx + j) % 3}"]), c)


def navy_kiosk(name, lo, hi, M, idx):
    """Free-standing navy signage kiosk with a screen on each long face (c_p1_a)."""
    c = "B2"
    (x0, y0), (x1, y1) = lo, hi
    bx(name, (x0, y0, Z_B2), (x1, y1, Z_B2 + 2.45), M["navy"], c, 0.012)
    along_x = (x1 - x0) >= (y1 - y0)
    if along_x:
        faces = (((x0 + x1) / 2, y0 - 0.006, (1, 0, 0)), ((x0 + x1) / 2, y1 + 0.006, (-1, 0, 0)))
        w = (x1 - x0) - 0.14
    else:
        faces = ((x0 - 0.006, (y0 + y1) / 2, (0, -1, 0)), (x1 + 0.006, (y0 + y1) / 2, (0, 1, 0)))
        w = (y1 - y0) - 0.14
    bx(name + "_cap", (x0 - 0.01, y0 - 0.01, Z_B2 + 2.45), (x1 + 0.01, y1 + 0.01, Z_B2 + 2.48), M["steel_brushed"], c, 0.004)
    bx(name + "_kick", (x0 - 0.008, y0 - 0.008, Z_B2), (x1 + 0.008, y1 + 0.008, Z_B2 + 0.09), M["steel_brushed"], c, 0.003)
    for j, (fx, fy, right) in enumerate(faces):
        rv = Vector(right)
        nrm = rv.cross(Vector((0, 0, 1)))
        hh = w * 1.6
        put(B.image_quad(f"{name}_scr{j}", (fx + nrm.x * 0.008, fy + nrm.y * 0.008, Z_B2 + 1.45), right, (0, 0, 1), w, hh, M[f"adp{(idx + j + 1) % 3}"]), c)
        for (du, dz, sw, sh) in ((0, hh / 2 + 0.02, w + 0.08, 0.04), (0, -hh / 2 - 0.02, w + 0.08, 0.04),
                                 (w / 2 + 0.02, 0, 0.04, hh), (-w / 2 - 0.02, 0, 0.04, hh)):
            ob = B.box(f"{name}_bz{j}", (0, 0, 0), (sw, 0.016, sh), M["screen_bezel"], 0.003)
            ob.location = (fx + rv.x * du + nrm.x * 0.006, fy + rv.y * du + nrm.y * 0.006, Z_B2 + 1.45 + dz)
            ob.rotation_euler = (0, 0, math.atan2(rv.y, rv.x))
            put(ob, c)


def build_vault(M):
    global VAULT
    VAULT = UV.build(sys.modules[__name__], M)
    build_central_wall_face(M)


def kiosk(name, lo, hi, M, idx):
    """Free-standing digital signage pillar: black body, white frame, louvred top band, a screen per face."""
    c = "B2"
    (x0, y0), (x1, y1) = lo, hi
    z0, zt = Z_B2, Z_B2 + 2.85
    bx(name + "_base", (x0 - 0.02, y0 - 0.02, z0), (x1 + 0.02, y1 + 0.02, z0 + 0.12), M["kiosk_white"], c, 0.005)
    bx(name + "_body", (x0, y0, z0 + 0.12), (x1, y1, zt - 0.40), M["kiosk_black"], c)
    bx(name + "_top", (x0 - 0.02, y0 - 0.02, zt - 0.40), (x1 + 0.02, y1 + 0.02, zt), M["kiosk_white"], c, 0.005)
    # corner posts
    for (cx, cy) in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        bx(name + "_post", (cx - 0.03, cy - 0.03, z0 + 0.12), (cx + 0.03, cy + 0.03, zt - 0.40), M["kiosk_white"], c, 0.004)
    faces = [((x0 + x1) / 2, y0 - 0.004, (1, 0, 0), x1 - x0), ((x0 + x1) / 2, y1 + 0.004, (-1, 0, 0), x1 - x0),
             (x0 - 0.004, (y0 + y1) / 2, (0, -1, 0), y1 - y0), (x1 + 0.004, (y0 + y1) / 2, (0, 1, 0), y1 - y0)]
    for j, (fx, fy, right, width) in enumerate(faces):
        w = min(0.56, width - 0.14)
        nrm = Vector(right).cross(Vector((0, 0, 1)))
        hgt = w * 1.78
        q = B.image_quad(f"{name}_scr{j}", (fx + nrm.x * 0.008, fy + nrm.y * 0.008, Z_B2 + 1.50), right, (0, 0, 1), w, hgt, M[f"adp{(idx + j) % 3}"])
        put(q, c)
        # glossy black bezel around the screen
        rv = Vector(right)
        for (du, dz, sw, sh) in ((0, hgt / 2 + 0.02, w + 0.08, 0.04), (0, -hgt / 2 - 0.02, w + 0.08, 0.04),
                                 (w / 2 + 0.02, 0, 0.04, hgt), (-w / 2 - 0.02, 0, 0.04, hgt)):
            ob = B.box(f"{name}_bz{j}", (0, 0, 0), (sw, 0.016, sh), M["screen_bezel"], 0.003)
            ob.location = (fx + rv.x * du + nrm.x * 0.006, fy + rv.y * du + nrm.y * 0.006, Z_B2 + 1.50 + dz)
            ob.rotation_euler = (0, 0, math.atan2(rv.y, rv.x))
            put(ob, c)
        # screws in the white top band, asset label near the base
        for su in (-width / 2 + 0.06, width / 2 - 0.06):
            for sz in (zt - 0.37, zt - 0.03):
                sc_ = B.cylinder(f"{name}_scw{j}", (0, 0, 0), 0.006, 0.004, M["steel"], segs=10)
                sc_.location = (fx + rv.x * su + nrm.x * 0.024, fy + rv.y * su + nrm.y * 0.024, sz)
                sc_.rotation_euler = (math.pi / 2, 0, math.atan2(rv.y, rv.x))
                put(sc_, c)
        if j < 2:
            put(B.image_quad(f"{name}_tag{j}", (fx + rv.x * (width / 2 - 0.16) + nrm.x * 0.002, fy + rv.y * (width / 2 - 0.16) + nrm.y * 0.002, Z_B2 + 0.32), right, (0, 0, 1), 0.15, 0.05, M["asset"]), c)
        # louvre grille in the top band
        for kk in range(6):
            zz = zt - 0.33 + kk * 0.045
            gx, gy = fx + nrm.x * 0.025, fy + nrm.y * 0.025
            hw = (width - 0.2) / 2
            if abs(right[0]) > 0:
                bx(f"{name}_lv{j}{kk}", (gx - hw, gy - 0.006, zz), (gx + hw, gy + 0.006, zz + 0.02), M["louver"], c)
            else:
                bx(f"{name}_lv{j}{kk}", (gx - 0.006, gy - hw, zz), (gx + 0.006, gy + hw, zz + 0.02), M["louver"], c)


def build_central_wall_face(M):
    """Face of the central wall under the LED screen: beige panels, arched openings, posters."""
    V = VAULT
    vx0, vx1 = V.x0, V.x1
    y = V.ye
    ztop = V.zs
    vrect("cw_face", (vx0, y), (vx1, y), Z_B2, ztop, M["beige"], "B2", (0, -1, 0), 2.4)
    bx("cw_skirt", (vx0, y - 0.03, Z_B2), (vx1, y, Z_B2 + 0.12), M["steel_brushed"], "B2")
    k = 0
    for x in np.arange(vx0 + 5.0, vx1 - 3, 9.0):
        w, hh = 1.6, 1.9
        pts = [(x - w / 2, Z_B2), (x + w / 2, Z_B2), (x + w / 2, Z_B2 + hh)]
        for i in range(1, 24):
            a = math.pi * i / 24
            pts.append((x + w / 2 * math.cos(a), Z_B2 + hh + w / 2 * math.sin(a)))
        pts.append((x - w / 2, Z_B2 + hh))
        verts = [(px_, y - 0.01, pz) for (px_, pz) in pts]
        me = bpy.data.meshes.new(f"arch_{k}")
        me.from_pydata(verts, [], [list(range(len(verts)))])
        me.update()
        box_uv(me)
        me.materials.append(M["door_dark"])
        coll("B2").objects.link(bpy.data.objects.new(f"arch_{k}", me))
        for i in range(len(pts) - 1):
            (ax, az), (bx_, bz) = pts[i], pts[i + 1]
            if az == Z_B2 and bz == Z_B2:
                continue
            ln = math.hypot(bx_ - ax, bz - az)
            ob = B.box(f"archf_{k}_{i}", (0, 0, 0), (ln + 0.01, 0.10, 0.12), M["panel_white"])
            ob.location = ((ax + bx_) / 2, y - 0.05, (az + bz) / 2)
            ob.rotation_euler = (0, -math.atan2(bz - az, bx_ - ax), 0)
            put(ob, "B2")
        for xa in (x + 3.0, x + 6.0):
            if xa < vx1 - 2.0:
                q = B.image_quad(f"cad_{k}_{xa:.0f}", (xa, y - 0.04, Z_B2 + 1.45), (1, 0, 0), (0, 0, 1), 1.03, 1.236, M[f"adp{(k + int(xa)) % 3}"])
                put(q, "B2")
                bx(f"cadf_{k}_{xa:.0f}", (xa - 0.56, y - 0.035, Z_B2 + 0.78), (xa + 0.56, y, Z_B2 + 2.12), M["steel"], "B2")
        k += 1


def floor_band(M):
    """Darker band of pavers along the line of the signage pillars (photos 2019-2024)."""
    V = VAULT
    y0, y1 = V.cy - 0.65, V.cy + 0.65
    rect("floor_band", 30.0, y0, 89.6, y1, Z_B2 + 0.0015, M["b2_floor_band"], "B2", True, 2.4)


def build_psd(M):
    """Platform screen doors (2021): white pocket cabinets between stainless door leaves with tall windows."""
    ye, yw = PLAN["edges_y"]["up"], PLAN["edges_y"]["lo"]
    px0, px1 = PLAN["plat_x"]
    for y, side, nm, length in ((ye, -1, "e", 197.0), (yw, 1, "w", 201.0)):
        y_in = y + side * 0.30
        xa = (px0 + px1) / 2 - length / 2
        car = length / 10
        doors = []
        for cidx in range(10):
            for d in range(4):
                doors.append(xa + cidx * car + car * (0.14 + d * 0.24))
        edges = [xa - 1.0] + [v for xd in doors for v in (xd, xd + 2.0)] + [xa + length + 1.0]
        k = 0
        for a, b in zip(edges[0::2], edges[1::2]):         # cabinets between door openings
            if b - a < 0.3:
                continue
            bx(f"psd_{nm}_cab_{k}", (a, y_in - 0.16, Z_B2), (b, y_in + 0.16, Z_B2 + 1.35), M["psd_panel"], "B2", 0.02)
            bx(f"psd_{nm}_cabtop_{k}", (a + 0.02, y_in - 0.165, Z_B2 + 1.30), (b - 0.02, y_in + 0.165, Z_B2 + 1.36), M["steel"], "B2", 0.01)
            # panel joints on the platform face of the cabinet
            yf_ = y_in + side * 0.161
            for xs_ in np.arange(a + 0.6, b - 0.3, 0.6):
                bx(f"psd_{nm}_seam_{k}_{xs_:.1f}", (xs_ - 0.002, min(yf_, yf_ + side * 0.002), Z_B2 + 0.12), (xs_ + 0.002, max(yf_, yf_ + side * 0.002), Z_B2 + 1.28), M["panel_grey"], "B2")
            bx(f"psd_{nm}_kick_{k}", (a + 0.01, min(yf_, yf_ + side * 0.004), Z_B2), (b - 0.01, max(yf_, yf_ + side * 0.004), Z_B2 + 0.10), M["steel_brushed"], "B2")
            # red car / door number label on the platform face
            if b - a > 0.8:
                yl = y_in + side * 0.166
                q = B.image_quad(f"psd_{nm}_lab_{k}", ((a + b) / 2, yl, Z_B2 + 1.08), (-side, 0, 0), (0, 0, 1), 0.36, 0.28, M["psd_label"])
                put(q, "B2")
            k += 1
        vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
        for i_, xd in enumerate(doors):                      # queue graphics on the floor (flat-ceiling part)
            if nm == "w" and vx0 - 2 < xd < vx1 + 2:
                continue
            for h_, off in ((0, -0.72), (1, 2.72)):          # one each side of the door, never overlapping
                xq = xd + off
                yq = y_in + side * (0.25 + 0.65)
                q = B.image_quad(f"psd_{nm}_q_{i_}{h_}", (xq, yq, Z_B2 + 0.003), (side, 0, 0), (0, side, 0), 1.0, 1.1, M[f"queue{(i_ + h_) % 2}"])
                put(q, "B2")
        for i_, xd in enumerate(doors):                      # closed door leaves
            for h_ in (0, 1):
                x0_ = xd + h_ * 1.0
                bx(f"psd_{nm}_leaf_{i_}{h_}", (x0_ + 0.01, y_in - 0.025, Z_B2 + 0.04), (x0_ + 0.99, y_in + 0.025, Z_B2 + 1.30), M["steel_brushed"], "B2", 0.004)
                bx(f"psd_{nm}_lwin_{i_}{h_}", (x0_ + 0.25, y_in - 0.028, Z_B2 + 0.35), (x0_ + 0.75, y_in + 0.028, Z_B2 + 1.18), M["psd_glass"], "B2", 0.01)
                # caution sticker on the platform face, rubber edge where the leaves meet
                ys_ = y_in + side * 0.0285
                put(B.image_quad(f"psd_{nm}_stk_{i_}{h_}", (x0_ + 0.5, ys_, Z_B2 + 0.24), (-side, 0, 0), (0, 0, 1), 0.36, 0.18, M["psd_sticker"]), "B2")
                xe_ = x0_ + (0.99 if h_ == 0 else 0.01)
                bx(f"psd_{nm}_rub_{i_}{h_}", (xe_ - 0.012, y_in - 0.03, Z_B2 + 0.05), (xe_ + 0.012, y_in + 0.03, Z_B2 + 1.29), M["handrail"], "B2", 0.004)
            # door-open lamp on the cabinet top next to the door
            put(B.cylinder(f"psd_{nm}_lamp_{i_}", (xd - 0.15, y_in, Z_B2 + 1.37), 0.025, 0.03, M["lamp_amber"], segs=12), "B2")


def build_tactile(M):
    ye, yw = PLAN["edges_y"]["up"], PLAN["edges_y"]["lo"]
    px0, px1 = PLAN["plat_x"]
    for y, side, nm in ((ye, -1, "e"), (yw, 1, "w")):
        yc = y + side * 1.0
        bx(f"tactile_{nm}", (px0 + 3, yc - 0.15, Z_B2), (px1 - 3, yc + 0.15, Z_B2 + 0.006), M["tactile"], "B2")


def build_b2_lights(M):
    px0, px1 = PLAN["plat_x"]
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    col = (1.0, 0.97, 0.92)
    k = 0
    for y in (16.4, 12.6, 8.4, -1.6, -6.4, -10.0):
        x = px0 + 2.5
        while x < px1 - 2.5:
            under_vault = vx0 - 1 < x < vx1 + 1 and y < 1.0
            if not under_vault:
                bx(f"ledh_{k}", (x - 2.75, y - 0.09, Z_B2_CEIL - 0.06), (x + 2.75, y + 0.09, Z_B2_CEIL), M["steel_brushed"], "B2_lights")
                bx(f"ledl_{k}", (x - 2.7, y - 0.06, Z_B2_CEIL - 0.065), (x + 2.7, y + 0.06, Z_B2_CEIL - 0.06), M["led_line"], "B2_lights")
                B.area_light(f"ledA_{k}", (x, y, Z_B2_CEIL - 0.07), 5.4, 0.12, 150.0, col, camera=False)
                put(bpy.data.objects[f"ledA_{k}"], "B2_lights")
                k += 1
            x += 5.6


def build_b2_signs(M):
    yw, tw = PLAN["edges_y"]["lo"], PLAN["track_wall_y"]["west"]
    te = PLAN["track_wall_y"]["east"]
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    # ads and station name boards on the track walls
    k = 0
    x = PLAN["plat_x"][0] + 6
    while x < PLAN["plat_x"][1] - 6:
        for y, rot, nm in ((tw + 0.03, (0, 1, 0), "w"), (te - 0.03, (0, -1, 0), "e")):
            right = (-1, 0, 0) if nm == "w" else (1, 0, 0)
            if k % 3 == 1:
                B.image_quad(f"nb_{nm}_{k}", (x, y, Z_B2 + 1.85), right, (0, 0, 1), 2.4, 0.95, M["name_board_" + nm])
                bx(f"nbf_{nm}_{k}", (x - 1.24, y - 0.04 if nm == "w" else y, Z_B2 + 1.33), (x + 1.24, y if nm == "w" else y + 0.04, Z_B2 + 2.37), M["steel"], "B2")
            else:
                B.image_quad(f"ad_{nm}_{k}", (x, y, Z_B2 + 1.8), right, (0, 0, 1), 3.6, 2.0, M[f"ad{k % 3}"])
                bx(f"adf_{nm}_{k}", (x - 1.85, y - 0.02 if nm == "w" else y, Z_B2 + 0.75), (x + 1.85, y if nm == "w" else y + 0.02, Z_B2 + 2.85), M["steel"], "B2")
        for ob in list(bpy.context.scene.collection.objects):
            put(ob, "B2")
        k += 1
        x += 9.0
    # departure boards with a clock, hanging over each half (wp_plat1, c_p2_depboard)
    for x in np.arange(PLAN["plat_x"][0] + 35, PLAN["plat_x"][1] - 15, 30.0):
        for y, img, nm in ((13.0, "dep1", "e"), (-6.0, "dep2", "w")):
            if vx0 - 3 < x < vx1 + 3 and y < 1:
                continue
            zc = Z_B2_CEIL - 0.55
            bx(f"dbbody_{nm}_{x:.0f}", (x - 0.12, y - 1.0, zc - 0.32), (x + 0.12, y + 1.0, zc + 0.32), M["panel_grey"], "B2", 0.01)
            for sgn_, im in ((-1, img), (1, img)):
                put(B.image_quad(f"db_{nm}_{x:.0f}_{sgn_}", (x + sgn_ * 0.125, y, zc), (0, sgn_, 0), (0, 0, 1), 1.9, 0.55, M[im]), "B2")
            bx(f"dbclk_{nm}_{x:.0f}", (x - 0.12, y - 1.55, zc - 0.27), (x + 0.12, y - 1.0, zc + 0.28), M["panel_grey"], "B2", 0.01)
            for sgn_ in (-1, 1):
                put(B.image_quad(f"dbclkf_{nm}_{x:.0f}_{sgn_}", (x + sgn_ * 0.125, y - 1.275, zc), (0, sgn_, 0), (0, 0, 1), 0.46, 0.46, M["clock"]), "B2")
            for dy in (-1.4, 0.8):
                bx(f"dbrod_{nm}_{x:.0f}_{dy}", (x - 0.012, y + dy - 0.012, zc + 0.3), (x + 0.012, y + dy + 0.012, Z_B2_CEIL), M["steel"], "B2")
    # hanging track number signs over the platform
    for x in np.arange(PLAN["plat_x"][0] + 20, PLAN["plat_x"][1] - 15, 30.0):
        for y, img, nm in ((15.0, "track1", "e"), (-4.5, "track2", "w")):
            if vx0 - 2 < x < vx1 + 2 and y < 1:
                continue          # no hanging signs under the arch (photos)
            zc = Z_B2_CEIL - 0.6
            ztop = Z_B2_CEIL
            B.image_quad(f"ts_{nm}_{x:.0f}", (x - 0.06, y, zc), (0, -1, 0), (0, 0, 1), 2.6, 0.52, M[img])
            B.image_quad(f"tsb_{nm}_{x:.0f}", (x + 0.06, y, zc), (0, 1, 0), (0, 0, 1), 2.6, 0.52, M["exit_sign"])
            bx(f"tsbody_{nm}_{x:.0f}", (x - 0.055, y - 1.32, zc - 0.28), (x + 0.055, y + 1.32, zc + 0.28), M["panel_grey"], "B2")
            for dy in (-1.0, 1.0):
                bx(f"tsrod_{nm}_{x:.0f}_{dy}", (x - 0.01, y + dy - 0.01, zc + 0.28), (x + 0.01, y + dy + 0.01, ztop), M["steel"], "B2")
        for ob in list(bpy.context.scene.collection.objects):
            put(ob, "B2")


# ------------------------------------------------------------- stairs, gates, exits

def flight(name, x_top, x_bot, y0, y1, z_top, z_bot, M, cname, landing=True, lower=None):
    rise = z_top - z_bot
    n = int(round(rise / 0.17))
    riser = rise / n
    run_total = abs(x_bot - x_top)
    sgn = 1 if x_bot > x_top else -1
    tread = 0.30
    land = max(0.0, run_total - n * tread) if landing else 0.0
    half = n // 2 if lower is None else n - lower
    x = x_top
    z = z_top
    path = [(x_top - sgn * 0.3, z_top), (x_top, z_top)]
    for i in range(n):
        if i == half and land > 0:
            xa, xb = sorted((x, x + sgn * land))
            bx(f"{name}_land", (xa, y0, z_bot), (xb, y1, z), M["stair"], cname)
            path += [(x, z), (x + sgn * land, z)]
            x += sgn * land
        z_step = z - riser
        xa, xb = sorted((x, x + sgn * tread))
        bx(f"{name}_s{i}", (xa, y0, z_bot), (xb, y1, z_step), M["stair"], cname)
        xn = x if sgn > 0 else x - 0.04
        bx(f"{name}_n{i}", (xn, y0 + 0.05, z_step - 0.004), (xn + 0.04, y1 - 0.05, z_step + 0.002), M["nosing"], cname)
        x += sgn * tread
        z = z_step
    path += [(x, z), (x + sgn * 0.3, z)]
    RAMPS.append((y0, y1, path))
    # handrails (sides + centre): they follow the flight (0.85 m over the nosings, level over the
    # landing, 0.3 m level run-outs at both ends); side rails sit on wall brackets, the centre rail
    # on posts standing on the treads
    rail_path = [(px_, pz + 0.85) for (px_, pz) in path]

    def tread_z(xq):
        """Top of the tread (or landing) under x."""
        zq = z_top
        for (xa_, za_), (xb_, zb_) in zip(path[:-1], path[1:]):
            lo_, hi_ = sorted((xa_, xb_))
            if lo_ - 1e-6 <= xq <= hi_ + 1e-6:
                if abs(zb_ - za_) < 1e-6:
                    return za_
                tq = (xq - xa_) / (xb_ - xa_)
                zl = za_ + tq * (zb_ - za_)
                # stepped: the tread under x is the step below the nosing line
                k_ = math.floor((zl - z_bot) / riser + 1e-6)
                return z_bot + k_ * riser
        return zq

    for yy, kind in ((y0 + 0.08, "wall"), ((y0 + y1) / 2, "post"), (y1 - 0.08, "wall")):
        tube_rail(f"{name}_hr{yy:.1f}", [(px_, yy, pz) for (px_, pz) in rail_path], 0.02, M["handrail_steel"], cname)
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rail_path[:-1], rail_path[1:]))
        step = 1.5 if kind == "post" else 1.2
        nsup = max(2, int(length / step) + 1)
        for j in range(nsup):
            # support points spread evenly along the rail (by length)
            target = 0.15 + (length - 0.3) * j / (nsup - 1)
            acc = 0.0
            for a, b in zip(rail_path[:-1], rail_path[1:]):
                seg = math.hypot(b[0] - a[0], b[1] - a[1])
                if acc + seg >= target:
                    tq = (target - acc) / seg
                    xq, zq = a[0] + tq * (b[0] - a[0]), a[1] + tq * (b[1] - a[1])
                    break
                acc += seg
            if kind == "post":
                zt = tread_z(xq)
                put(B.cylinder(f"{name}_post{j}", (xq, yy, (zt + zq) / 2), 0.022, zq - zt, M["handrail_steel"], segs=12), cname)
                put(B.cylinder(f"{name}_base{j}", (xq, yy, zt + 0.004), 0.06, 0.008, M["handrail_steel"], segs=16), cname)
            else:
                yw = y0 - 0.0 if yy < (y0 + y1) / 2 else y1
                ya_, yb_ = sorted((yw, yy))
                bx(f"{name}_br{yy:.1f}_{j}", (xq - 0.012, ya_, zq - 0.075), (xq + 0.012, yb_, zq - 0.055), M["handrail_steel"], cname)
                bx(f"{name}_brv{yy:.1f}_{j}", (xq - 0.008, yy - 0.008, zq - 0.075), (xq + 0.008, yy + 0.008, zq - 0.018), M["handrail_steel"], cname)
                plate_y = (yw, yw + 0.01) if yy > yw else (yw - 0.01, yw)
                bx(f"{name}_brp{yy:.1f}_{j}", (xq - 0.04, plate_y[0], zq - 0.11), (xq + 0.04, plate_y[1], zq - 0.02), M["handrail_steel"], cname)


def tube_rail(name, pts, r, mat, cname):
    """Round rail along a polyline, with ball joints at the bends and end caps turned down."""
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        if math.dist(a, b) < 1e-4:
            continue
        slope_rail(f"{name}_{i}", a, b, r, mat, cname)
    for i, p in enumerate(pts):
        put(B.sphere(f"{name}_j{i}", p, r, mat, segs=12), cname)
    # ends bent down by 0.1 m (no open tube ends)
    for i, p in ((0, pts[0]), (1, pts[-1])):
        slope_rail(f"{name}_end{i}", p, (p[0], p[1], p[2] - 0.1), r, mat, cname)
        put(B.sphere(f"{name}_endb{i}", (p[0], p[1], p[2] - 0.1), r, mat, segs=12), cname)


def build_stairs(M):
    for k, st in PLAN["stairs"].items():
        y0, y1 = st["y0"], st["y1"]
        if k not in ("J", "I"):
            flight(f"st{k}", st["x_top"], st["x_bot"], y0, y1, Z_B1, Z_B2, M, "Stairs")
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        # stairwell side walls and the head wall over the stair foot
        if k == "J":
            stair_j(M)        # under the arch
            continue
        if k == "I":
            stair_esc("I", M, under_vault=False)
            continue
        for yy, nm in ((y0 - 0.15, "a"), (y1, "b")):
            bx(f"st{k}_wall{nm}", (xa, yy, Z_B2), (xb, yy + 0.15, Z_B1), M["wall_tile"], "Stairs")
        xh = st["x_bot"]
        ha, hb = sorted((xh, xh + (0.3 if st["x_bot"] > st["x_top"] else -0.3)))
        bx(f"st{k}_head", (ha, y0, Z_B2_CEIL), (hb, y1, Z_B1 - 0.35), M["panel_white"], "Stairs")
        # tactile warning strips at the top (B1) and the foot (B2)
        sg = 1 if st["x_bot"] > st["x_top"] else -1
        for xx, zz in ((st["x_top"] - sg * 0.3, Z_B1), (st["x_bot"] + sg * 0.3, Z_B2)):
            lo, hi = sorted((xx, xx - sg * 0.3 if zz == Z_B1 else xx + sg * 0.3))
            bx(f"st{k}_tw{zz:.0f}", (lo, y0 + 0.1, zz), (hi, y1 - 0.1, zz + 0.006), M["tactile"], "Stairs")
        # glass balustrade around the B1 opening (not on the entry side)
        for yy in (y0, y1):
            bx(f"st{k}_bal{yy:.1f}", (xa, yy - 0.02, Z_B1), (xb, yy + 0.02, Z_B1 + 1.1), M["glass_rail"], "B1")
            bx(f"st{k}_balt{yy:.1f}", (xa, yy - 0.04, Z_B1 + 1.1), (xb, yy + 0.04, Z_B1 + 1.14), M["steel"], "B1")
        xe = st["x_bot"]
        bx(f"st{k}_bale", (xe - 0.02, y0, Z_B1), (xe + 0.02, y1, Z_B1 + 1.1), M["glass_rail"], "B1")



def incline_prism(name, pts_xz, y0, y1, mat, cname):
    """Extrude a polygon given in the x-z plane between y0 and y1."""
    poly = [[(px_, pz) for px_, pz in pts_xz]]
    ob = prism(name, poly, y0, y1, mat, cname)
    # prism() extrudes along z; swap axes: (x, y_poly=z, z_extrude=y) -> (x, y, z)
    me = ob.data
    for v in me.vertices:
        x, zz, yy = v.co.x, v.co.y, v.co.z
        v.co = (x, yy, zz)
    for p in me.polygons:
        p.flip()
    me.update()
    return ob


def slope_rail(name, p0, p1, r, mat, cname, axis_box=None):
    """Round bar between two points."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    if axis_box:
        ob = B.box(name, (0, 0, 0), (d.length, axis_box[0], axis_box[1]), mat, 0.01)
    else:
        ob = B.cylinder(name, (0, 0, 0), r, d.length, mat, axis="X", segs=16)
    ob.location = (p0 + p1) / 2
    ob.rotation_euler = Vector((1, 0, 0)).rotation_difference(d.normalized()).to_euler()
    return put(ob, cname)


def escalator(name, x_top, x_bot, y0, y1, M, cname):
    """Escalator between B1 (x_top) and B2 (x_bot): steps, glass balustrades, black handrails."""
    sgn = 1 if x_bot > x_top else -1
    flat = 1.9
    xa, xb = x_bot - sgn * flat, x_top + sgn * flat      # ends of the incline
    rise = Z_B1 - Z_B2
    run = abs(xb - xa)
    n = int(round(run / 0.40))
    tread = run / n
    dz = rise / n
    # truss mass under the steps
    pts = [(x_bot, Z_B2 - 0.3), (x_bot, Z_B2), (xa, Z_B2), (xb, Z_B1), (x_top, Z_B1), (x_top, Z_B2 - 0.3)]
    incline_prism(name + "_truss", pts, y0, y1, M["esc_clad"], cname)
    RAMPS.append((y0 + 0.28, y1 - 0.28, [(x_bot + sgn * 0.3, Z_B2), (xa, Z_B2), (xb, Z_B1), (x_top - sgn * 0.3, Z_B1)]))
    # steps (sit on top of the truss line)
    wy0, wy1 = y0 + 0.28, y1 - 0.28
    for i in range(n):
        x0 = xa - sgn * i * tread
        x1 = x0 - sgn * tread
        zt = Z_B2 + (i + 1) * dz
        lo, hi = sorted((x0, x1))
        bx(f"{name}_st{i}", (lo, wy0, zt - dz - 0.02), (hi, wy1, zt + 0.02), M["esc_step"], cname)
        e = lo if sgn < 0 else hi - 0.05
        bx(f"{name}_ye{i}", (e, wy0, zt + 0.02), (e + 0.05, wy1, zt + 0.025), M["nosing"], cname)
    for xx, zz in ((x_bot, Z_B2), (x_top, Z_B1)):
        lo, hi = sorted((xx, xx - sgn * flat))
        bx(f"{name}_plate{zz:.0f}", (lo, wy0, zz), (hi, wy1, zz + 0.025), M["steel_brushed"], cname)
    # balustrades: glass panel + black handrail following flat-incline-flat
    path = [(x_bot - sgn * 0.3, Z_B2), (xa, Z_B2), (xb, Z_B1), (x_top + sgn * 0.3, Z_B1)]
    for yy, side in ((y0 + 0.18, "w"), (y1 - 0.18, "e")):
        for j in range(len(path) - 1):
            (ax, az), (bx_, bz) = path[j], path[j + 1]
            verts = [(ax, yy, az + 0.12), (bx_, yy, bz + 0.12), (bx_, yy, bz + 0.92), (ax, yy, az + 0.92)]
            q = B.mesh_obj(f"{name}_gl{side}{j}", verts, [(0, 1, 2, 3)], M["glass_rail"])
            put(q, cname)
            slope_rail(f"{name}_hr{side}{j}", (ax, yy, az + 0.96), (bx_, yy, bz + 0.96), 0, M["handrail"], cname, axis_box=(0.09, 0.05))
            slope_rail(f"{name}_dk{side}{j}", (ax, yy + (0.08 if side == "w" else -0.08), az + 0.10), (bx_, yy + (0.08 if side == "w" else -0.08), bz + 0.10), 0, M["steel_brushed"], cname, axis_box=(0.18, 0.04))


def parapet(name, x_top, x_bot, y0, y1, M, cname, h=1.05):
    pts = [(x_bot, Z_B2 - 0.2), (x_bot, Z_B2 + h), (x_top, Z_B1 + h), (x_top, Z_B2 - 0.2)]
    incline_prism(name, pts, y0, y1, M["parapet"], cname)
    bx(name + "_cap", (min(x_bot, x_top), y0 - 0.01, Z_B2 - 0.2), (min(x_bot, x_top) + 0.01, y1 + 0.01, Z_B2), M["parapet"], cname)
    slope_rail(name + "_cap", (x_bot, (y0 + y1) / 2, Z_B2 + h + 0.02), (x_top, (y0 + y1) / 2, Z_B1 + h + 0.02), 0, M["steel"], cname, axis_box=(y1 - y0 + 0.04, 0.04))


def stair_esc(k, M, portal_sign=True, under_vault=True):
    """Stair + escalator in one opening (stairs J and I): stair on the west, escalator on the east,
    black portal frame with the track direction sign at the foot (photos 2019-2024)."""
    st = PLAN["stairs"][k]
    c = "Stairs"
    x_top, x_bot = st["x_top"], st["x_bot"]
    sgn = 1 if x_bot > x_top else -1          # +1: descends toward +x
    y0, y1 = st["y0"], st["y1"]
    ys = y1 - 1.85                     # stair | escalator split
    flight(f"st{k}s", x_top, x_bot, y0 + 0.2, ys - 0.1, Z_B1, Z_B2, M, c, lower=14)
    escalator(f"esc{k}", x_top, x_bot, ys, y1 - 0.1, M, c)
    if under_vault:
        parapet(f"st{k}_pw", x_top, x_bot, y0 - 0.05, y0 + 0.2, M, c)
        parapet(f"st{k}_pe", x_top, x_bot, y1 - 0.1, y1 + 0.15, M, c, h=1.15)
    else:
        xa, xb = sorted((x_top, x_bot))
        for yy, nm in ((y0 - 0.15, "a"), (y1, "b")):
            bx(f"st{k}_wall{nm}", (xa, yy, Z_B2), (xb, yy + 0.15, Z_B1), M["track_tile"], c)
        ha, hb = sorted((x_bot, x_bot + sgn * 0.3))
        bx(f"st{k}_head", (ha, y0, Z_B2_CEIL), (hb, y1, Z_B1 - 0.35), M["panel_white"], c)
    bx(f"st{k}_div", (min(x_top, x_bot), ys - 0.1, Z_B2), (max(x_top, x_bot), ys + 0.02, Z_B2 + 0.2), M["steel_brushed"], c)
    # portal frame at the foot
    xp = x_bot + sgn * 0.45            # just beyond the foot, on the platform
    pa, pb = y0 - 0.45, y1 + 0.45
    for yy in (pa, pb - 0.38):
        bx(f"portal{k}_post{yy:.0f}", (xp - 0.19, yy, Z_B2), (xp + 0.19, yy + 0.38, Z_B2 + 3.15), M["portal"], c, 0.006)
    bx(f"portal{k}_beam", (xp - 0.19, pa, Z_B2 + 2.55), (xp + 0.19, pb, Z_B2 + 3.15), M["portal"], c, 0.006)
    w = pb - pa - 0.9
    face = xp - sgn * 0.195            # faces up the stair
    right = (0, -sgn, 0)
    if portal_sign:
        put(B.image_quad(f"portal{k}_sign_s", (face, (pa + pb) / 2, Z_B2 + 2.85), right, (0, 0, 1), w, w / 14.0, M["portal_sign"]), c)
    put(B.image_quad(f"portal{k}_sign_n", (xp + sgn * 0.195, (pa + pb) / 2, Z_B2 + 2.85), (0, sgn, 0), (0, 0, 1), 3.2, 3.2 / 5.0, M["exit_sign"]), c)
    # tactile warning strips at the foot and the top; arrow decals at both ends of the landing
    for xx, zz in ((x_bot + sgn * 0.3, Z_B2), (x_top - sgn * 0.3, Z_B1)):
        lo, hi = sorted((xx, xx + sgn * 0.3 if zz == Z_B2 else xx - sgn * 0.3))
        bx(f"st{k}_tw{zz:.0f}", (lo, y0 + 0.2, zz), (hi, ys - 0.1, zz + 0.006), M["tactile"], c)
    n = int(round((Z_B1 - Z_B2) / 0.17))
    riser = (Z_B1 - Z_B2) / n
    z_land = Z_B2 + 14 * riser
    x_land = x_bot - sgn * 14 * 0.30          # edge of the landing (top of the lower flight)
    wst = (ys - 0.1) - (y0 + 0.2)
    for j_ in range(4):
        yy = y0 + 0.2 + wst * (j_ + 0.5) / 4
        mat = M["decal_up"] if j_ < 2 else M["decal_down"]
        q = B.image_quad(f"st{k}_dec{j_}", (x_land - sgn * 0.32, yy, z_land + 0.003), (0, 1, 0) if sgn > 0 else (0, -1, 0), (-sgn, 0, 0), 0.42, 0.42, mat)
        put(q, c)


def stair_j(M):
    stair_esc("J", M, under_vault=True)


def guide_lines(M):
    """Yellow tactile guide lines on the platform at the foot of stair J (2022 photo)."""
    st = PLAN["stairs"]["J"]
    c = "B2"
    yw = PLAN["edges_y"]["lo"]
    xb = st["x_bot"] - 0.95
    # where lines cross, the later block sits 0.5 mm higher so the tops never coincide
    bx("gl_j_long", (40.0, yw + 2.05, Z_B2), (xb, yw + 2.35, Z_B2 + 0.006), M["tactile_bar"], c)
    bx("gl_j_cross", (xb - 5.3, yw + 1.0, Z_B2), (xb - 5.0, PLAN["vault"]["y_east"] - 0.6, Z_B2 + 0.0065), M["tactile"], c)
    bx("gl_j_foot", (xb - 0.3, yw + 2.05, Z_B2), (xb, st["y1"] + 0.4, Z_B2 + 0.007), M["tactile"], c)


def build_gates(M):
    """Automatic ticket gates as in the 2023 photos: brushed stainless bodies, black top with a blue
    IC reader, yellow flap doors, green arrow / red cross indicators at both ends; black fascia band
    with the gate name above the north gate line."""
    k = 0
    unpaid_minus = {"北改札": True, "中北東改札": True, "中北西改札": False, "南改札": False, "連絡改札": False}
    for name, pl in PLAN["gates"].items():
        pts = [Vector((p[0], p[1], 0)) for p in pl]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            L = d.length
            t = d.normalized()
            nrm = Vector((-t.y, t.x, 0))          # along the passage
            n = max(1, int(L / 0.95))
            ang = math.atan2(nrm.y, nrm.x)
            ca, sa = math.cos(ang), math.sin(ang)
            gsg = 1 if unpaid_minus[name] else -1          # green arrows face the unpaid side
            for i in range(n + 1):
                p = a + t * (L * i / n)
                parts = [
                    ((0, 0, 0.49), (1.70, 0.27, 0.98), M["gate_side"], 0.06),
                    ((0, 0, 0.99), (1.62, 0.28, 0.03), M["gate_top"], 0.012),
                    ((-0.55 * gsg, 0, 1.008), (0.24, 0.18, 0.008), M["gate_blue"], 0.004),
                    ((0.55 * gsg, 0, 1.008), (0.24, 0.18, 0.008), M["gate_blue"], 0.004),
                    ((-0.852 * gsg, 0, 0.78), (0.006, 0.12, 0.12), M["gate_green"], 0),
                    ((0.852 * gsg, 0, 0.78), (0.006, 0.12, 0.12), M["gate_red"] if i % 3 == 0 else M["gate_green"], 0),
                    ((-0.852 * gsg, 0, 0.60), (0.006, 0.14, 0.16), M["gate_panel"], 0),
                    ((0.852 * gsg, 0, 0.60), (0.006, 0.14, 0.16), M["gate_panel"], 0),
                    ((0.05, 0.215, 0.74), (0.36, 0.16, 0.30), M["gate_yellow"], 0.04),
                    ((0.05, -0.215, 0.74), (0.36, 0.16, 0.30), M["gate_yellow"], 0.04),
                ]
                if i in (0, n) and False:
                    pass
                for j_, (off, size, mat, bev) in enumerate(parts):
                    if j_ >= 8 and (i == 0 and j_ == 9 or i == n and j_ == 8):
                        continue          # no flap on the outer side of the end cabinets
                    if j_ in (2, 3, 4, 5):
                        continue          # replaced by textured parts below
                    ob = B.box(f"gate_{k}_{j_}", (0, 0, 0), size, mat, bev)
                    ob.location = (p.x + ca * off[0] - sa * off[1], p.y + sa * off[0] + ca * off[1], Z_B1 + off[2])
                    ob.rotation_euler = (0, 0, ang)
                    put(ob, "B1")

                def at(u, v, z):
                    return (p.x + ca * u - sa * v, p.y + sa * u + ca * v, Z_B1 + z)
                fwd = (ca, sa, 0)
                side_v = (-sa, ca, 0)
                for su in (-1, 1):
                    u = su * 0.55 * gsg
                    # IC reader (textured, glowing) and the ticket slot / small display on the top
                    put(B.image_quad(f"gate_{k}_ic{su}", at(u, 0, 1.0065), fwd if su * gsg < 0 else (-ca, -sa, 0), side_v if su * gsg < 0 else (sa, -ca, 0), 0.15, 0.20, M["ic_reader"]), "B1")
                    sl = B.box(f"gate_{k}_slot{su}", (0, 0, 0), (0.012, 0.10, 0.004), M["handrail"])
                    sl.location = at(su * 0.25 * gsg, 0, 1.006)
                    sl.rotation_euler = (0, 0, ang)
                    put(sl, "B1")
                    put(B.image_quad(f"gate_{k}_lcd{su}", at(su * 0.32 * gsg, 0.0, 1.0068), fwd, side_v, 0.06, 0.10, M["dep1"]), "B1")
                    # end-face indicator: arrow on the entry side, arrow or cross on the exit side
                    ok = su < 0 or i % 3 != 0
                    nrmv = (su * gsg * ca, su * gsg * sa, 0)
                    rgt = (-nrmv[1], nrmv[0], 0)
                    put(B.image_quad(f"gate_{k}_ind{su}", at(su * 0.856 * gsg, 0, 0.78), rgt, (0, 0, 1), 0.12, 0.12, M["gate_ok" if ok else "gate_ng"]), "B1")
                    put(B.image_quad(f"gate_{k}_pnl{su}", at(su * 0.856 * gsg, 0, 0.60), rgt, (0, 0, 1), 0.14, 0.16, M["ic_reader"]), "B1")
                k += 1
            if name == "北改札":
                # black fascia band over the gate line with the gate name on the unpaid side
                sgn = -1 if unpaid_minus[name] else 1
                mid = (a + b) / 2
                fas = B.box(f"fascia_{k}", (0, 0, 0), (0.5, L + 0.6, 0.85), M["portal"], 0.01)
                fas.location = (mid.x, mid.y, Z_B1 + H_B1 - 0.43)      # hangs just below the ceiling
                fas.rotation_euler = (0, 0, ang)
                put(fas, "B1")
                if L > 6:
                    face = mid + nrm * (sgn * 0.26)
                    right = t * (1 if sgn < 0 else -1)
                    for off_, img, w_ in ((-0.25, "gate_fascia_n", 6.4),):
                        q = B.image_quad(f"fascia_sign_{k}", (face.x, face.y, Z_B1 + H_B1 - 0.43), (right.x, right.y, 0), (0, 0, 1), w_, w_ / 9.0, M[img])
                        put(q, "B1")


def build_exits(M):
    """Exit stairwells up to the street and the street itself (umeda_street)."""
    UST.build(sys.modules[__name__], M)


def in_exit_cut(x, y, z):
    for F, lo, hi in EXIT_CUTS:
        p = F.inverted() @ Vector((x, y, z))
        if lo[0] <= p.x <= hi[0] and lo[1] <= p.y <= hi[1] and lo[2] - 0.2 <= p.z <= hi[2]:
            return True
    return False


def build_b1_lights(M):
    """Round downlights on a 3 m grid over the walkable concourse (gate area photos)."""
    import cv2
    walk = [np.array(poly[0], np.float32).reshape(-1, 1, 2) for poly in PLAN["b1_walk"]]
    cr = PLAN["b1_corridor"]
    xs = np.arange(-140, 140, 3.0)
    ys = np.arange(-50, 50, 3.0)
    k = 0
    for x in xs:
        for y in ys:
            if cr["x0"] < x < cr["x1"] and cr["y_s"] < y < cr["y_n"]:
                continue        # the vaulted corridor has its own light rings
            if any(cv2.pointPolygonTest(r, (float(x), float(y)), False) >= 0 for r in walk) and not in_exit_cut(x, y, H_B1):
                put(B.cylinder(f"b1dl_{k}", (x, y, H_B1 - 0.004), 0.09, 0.01, M["downlight"], segs=20), "B1_lights")
                put(B.cylinder(f"b1dlr_{k}", (x, y, H_B1 - 0.002), 0.11, 0.006, M["steel"], segs=20), "B1_lights")
                ob = B.area_light(f"b1A_{k}", (x, y, H_B1 - 0.02), 0.2, 0.2, 85.0, (1.0, 0.96, 0.9), camera=False)
                ob.data.spread = math.radians(150)
                put(ob, "B1_lights")
                k += 1
    print("B1 lights", k)


CONT_W = -132.4      # the concourse carries on west of the map (towards the JR / Hankyu stations)
CONT_W_END = -162.0  # where it reaches the gate line of the next station
CONT_W_Y = (-21.575, 19.144)


def _skip_perimeter(level, x, y):
    if level == "B1":
        for F, lo, hi in EXIT_CUTS:
            p = F.inverted() @ Vector((x, y, 0.0))
            if -1.5 <= p.x <= hi[0] and lo[1] - 0.6 <= p.y <= hi[1] + 0.6:
                return True                  # foot of an exit stair
        V = PLAN["vault"]
        if V["x0"] - 0.6 < x < V["x1"] + 0.6 and V["y_west"] - 0.6 < y < V["y_east"] + 0.6:
            return True                      # the void over the arch (balustraded)
        if x < CONT_W + 0.3 and CONT_W_Y[0] - 0.3 < y < CONT_W_Y[1] + 0.3:
            return True                      # carries on (build_continuation)
    else:
        e = PLAN["edges_y"]
        if y > e["up"] - 0.7 or y < e["lo"] + 0.7:
            return True                      # platform edges and track sides
    return False


def build_perimeter(M):
    """Outer walls of the concourse and the platform level along the floor outline wherever the
    map draws no wall (the vectorised walls are partitions and rooms, not the outline), so the
    station ends in proper walls instead of the model just stopping."""
    import cv2
    from scipy import ndimage
    res = 0.1
    X0, Y0 = -150.0, -70.0
    W, H = int(300 / res), int(140 / res)

    def mask(polys):
        m = np.zeros((H, W), np.uint8)
        for poly in polys:
            cv2.fillPoly(m, [np.array([[(x - X0) / res, (y - Y0) / res] for x, y in poly[0]], np.int32)], 1)
        return m.astype(bool)
    k = 0
    for level, fl, wl, z0, z1, mat in (("B1", "b1_floor", "b1_walls", Z_B1, Z_B1 + H_B1 + 0.04, M["b1_wall"]),
                                       ("B2", "b2_floor", "b2_walls", Z_B2, Z_B2_CEIL + 0.04, M["wall_white"])):
        near = ndimage.binary_dilation(mask(PLAN[wl]), iterations=4)
        inside = mask(PLAN[fl])
        for poly in PLAN[fl]:
            ring = np.array(poly[0] + [poly[0][0]], float)
            for a, b in zip(ring[:-1], ring[1:]):
                L = float(np.linalg.norm(b - a))
                if L < 0.05:
                    continue
                d = (b - a) / L
                nrm = np.array((-d[1], d[0]))
                mid = (a + b) / 2 + nrm * 0.3
                ix, iy = int((mid[0] - X0) / res), int((mid[1] - Y0) / res)
                out = -nrm if (0 <= iy < H and 0 <= ix < W and inside[iy, ix]) else nrm
                n = max(2, int(L / 0.2) + 1)
                ts = np.linspace(0, 1, n)
                free = []
                for t in ts:
                    p = a + (b - a) * t
                    ix, iy = int((p[0] - X0) / res), int((p[1] - Y0) / res)
                    cov = 0 <= iy < H and 0 <= ix < W and near[iy, ix]
                    free.append(not cov and not _skip_perimeter(level, p[0], p[1]))
                i = 0
                while i < n:
                    if not free[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and free[j + 1]:
                        j += 1
                    p0, p1 = a + (b - a) * ts[i], a + (b - a) * ts[j]
                    if np.linalg.norm(p1 - p0) >= 0.4:
                        q = [tuple(p0), tuple(p1), tuple(p1 + out * 0.2), tuple(p0 + out * 0.2)]
                        prism(f"perim_{level}_{k}", [q], z0, z1, mat, level, uv_scale=2.4)
                        q2 = [tuple(p0 - out * 0.012), tuple(p1 - out * 0.012), tuple(p1), tuple(p0)]
                        prism(f"perim_sk_{level}_{k}", [q2], z0, z0 + 0.1, M["steel_brushed"], level)
                        k += 1
                    i = j + 1
    print("perimeter walls", k)


def build_continuation(M):
    """West of the map the concourse goes on towards the JR and Hankyu stations: the same floor,
    ceiling, columns and walls for another 30 m, to the gate line of the next station (its gates
    are shut to the walker; the hall beyond is seen through them)."""
    c = "B1"
    ya, yb = CONT_W_Y
    xa, xb = CONT_W_END - 22.0, CONT_W + 0.05
    poly = [[(xa, ya), (xb, ya), (xb, yb), (xa, yb)]]
    prism("cont_floor", poly, Z_B1 - 0.35, Z_B1, M["b1_floor"], c, uv_scale=2.4)
    prism("cont_ceil", poly, Z_B1 + H_B1, Z_B1 + H_B1 + 0.05, M["ceil"], "B1_ceiling", uv_scale=1.2)
    for yy, nm in ((ya - 0.2, "s"), (yb, "n")):
        bx(f"cont_wall_{nm}", (xa, yy, Z_B1), (xb, yy + 0.2, Z_B1 + H_B1 + 0.04), M["b1_wall"], c)
        bx(f"cont_sk_{nm}", (xa, yy + (0.19 if nm == "s" else -0.002), Z_B1), (xb, yy + (0.212 if nm == "s" else 0.01), Z_B1 + 0.1), M["steel_brushed"], c)
    bx("cont_end", (xa - 0.2, ya - 0.2, Z_B1), (xa, yb + 0.2, Z_B1 + H_B1 + 0.04), M["b1_wall"], c)
    # columns on the concourse grid
    for x in np.arange(CONT_W - 7.5, xa + 2, -7.5):
        for y in np.arange(ya + 6.0, yb - 3.0, 7.5):
            prism(f"cont_col_{x:.0f}_{y:.0f}", [[(x - 0.45, y - 0.45), (x + 0.45, y - 0.45), (x + 0.45, y + 0.45), (x - 0.45, y + 0.45)]],
                  Z_B1 + 0.15, Z_B1 + H_B1, M["column"], c)
            prism(f"cont_colsk_{x:.0f}_{y:.0f}", [[(x - 0.46, y - 0.46), (x + 0.46, y - 0.46), (x + 0.46, y + 0.46), (x - 0.46, y + 0.46)]],
                  Z_B1, Z_B1 + 0.15, M["steel_brushed"], c)
    # the next station's gate line, shut (flaps closed) across the whole hall
    PLAN["gates"]["連絡改札"] = [[CONT_W_END, ya + 0.4], [CONT_W_END, yb - 0.4]]
    # signs on the way: hanging boards pointing on to the JR / Hankyu lines
    for x in (CONT_W - 4.0, CONT_W - 18.0):
        hang_sign(f"cont_sign_{x:.0f}", x, (ya + yb) / 2, M["exit_b1"], M["exit_b1"], c, w=4.2, h=0.62)


def build_fill_lights(M):
    """Light every floor the walker can reach. On the 3 m downlight grid, wherever there is floor
    at a level with a ceiling over it and no light within 3.2 m (side corridors, platform ends,
    rooms in the island structure, exit passages), a downlight goes on that ceiling."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    verts, tris, off = [], [], 0
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.data.polygons:
            continue
        if any(c.name.endswith("_lights") for c in ob.users_collection):
            continue
        eo = ob.evaluated_get(dg)
        me = eo.to_mesh()
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.empty(n * 3, np.float64)
        me.vertices.foreach_get("co", co)
        mw = np.array(ob.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        t = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        verts.append(co)
        tris.append(t.reshape(-1, 3) + off)
        off += n
        eo.to_mesh_clear()
    tree = BVHTree.FromPolygons(np.vstack(verts).tolist(), np.vstack(tris).tolist())
    lights = [ob.matrix_world.translation.copy() for ob in bpy.data.objects if ob.type == "LIGHT"]
    k = 0
    for z0, cname in ((Z_B1, "B1_lights"), (Z_B2, "B2_lights")):
        for x in np.arange(-185, 140, 3.0):
            for y in np.arange(-50, 50, 3.0):
                f = tree.ray_cast(Vector((x, y, z0 + 1.0)), Vector((0, 0, -1)), 1.3)
                if f[0] is None or f[1].z < 0.7 or abs(f[0].z - z0) > 0.25:
                    continue
                c = tree.ray_cast(f[0] + Vector((0, 0, 0.05)), Vector((0, 0, 1)), 8.0)
                if c[0] is None or c[1].z > -0.5 or not 2.1 < c[0].z - f[0].z < 7.0:
                    continue
                zc = c[0].z
                if any((L.x - x) ** 2 + (L.y - y) ** 2 < 3.2 ** 2 and abs(L.z - zc) < 2.0 for L in lights):
                    continue
                put(B.cylinder(f"filldl_{k}", (x, y, zc - 0.004), 0.09, 0.01, M["downlight"], segs=20), cname)
                put(B.cylinder(f"filldlr_{k}", (x, y, zc - 0.002), 0.11, 0.006, M["steel"], segs=20), cname)
                ob = B.area_light(f"fillA_{k}", (x, y, zc - 0.02), 0.2, 0.2, 85.0, (1.0, 0.96, 0.9), camera=False)
                ob.data.spread = math.radians(150)
                put(ob, cname)
                lights.append(Vector((x, y, zc - 0.02)))
                k += 1
    print("fill lights", k)


def build_details(M):
    """Small ceiling fixtures seen in every photo: CCTV domes, speakers, green emergency-exit signs."""
    import cv2
    rng = np.random.default_rng(12)
    walk2 = [np.array(poly[0], np.float32).reshape(-1, 1, 2) for poly in PLAN["b2_walk"]]
    walk1 = [np.array(poly[0], np.float32).reshape(-1, 1, 2) for poly in PLAN["b1_walk"]]
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    k = 0
    for (x, y) in [(x, y) for x in np.arange(-100, 110, 11.0) for y in (4.0, 13.0, -3.0, -8.5)]:
        if vx0 - 1 < x < vx1 + 1 and y < 1.5:
            continue
        x += rng.uniform(-1.5, 1.5)
        if not any(cv2.pointPolygonTest(r, (float(x), float(y)), False) >= 0 for r in walk2):
            continue
        z = Z_B2_CEIL
        if k % 2 == 0:
            put(B.sphere(f"cctv_{k}", (x, y, z - 0.07), 0.075, M["cctv"], scale=(1, 1, 0.9)), "B2")
            put(B.cylinder(f"cctvb_{k}", (x, y, z - 0.02), 0.09, 0.04, M["panel_white"], segs=24), "B2")
        else:
            bx(f"spk_{k}", (x - 0.14, y - 0.14, z - 0.12), (x + 0.14, y + 0.14, z), M["panel_white"], "B2", 0.01)
            bx(f"spkg_{k}", (x - 0.11, y - 0.11, z - 0.121), (x + 0.11, y + 0.11, z - 0.119), M["louver"], "B2")
        k += 1
    for (x, y) in [(x, y) for x in np.arange(-130, 120, 9.0) for y in (-6.0, 6.0, -15.0)]:
        x += rng.uniform(-1.0, 1.0)
        if not any(cv2.pointPolygonTest(r, (float(x), float(y)), False) >= 0 for r in walk1):
            continue
        cr = PLAN["b1_corridor"]
        if cr["x0"] < x < cr["x1"] and cr["y_s"] < y < cr["y_n"]:
            continue
        put(B.sphere(f"cctv1_{k}", (x, y, H_B1 - 0.07), 0.075, M["cctv"], scale=(1, 1, 0.9)), "B1")
        put(B.cylinder(f"cctvb1_{k}", (x, y, H_B1 - 0.02), 0.09, 0.04, M["panel_white"], segs=24), "B1")
        k += 1
    # green emergency-exit signs over each stair foot (B2) and top (B1)
    for kk, st in PLAN["stairs"].items():
        sg = 1 if st["x_bot"] > st["x_top"] else -1
        yc = (st["y0"] + st["y1"]) / 2
        for xx, zz, face in ((st["x_bot"] + sg * 0.9, Z_B2_CEIL if not (vx0 < st["x_bot"] < vx1 and yc < 1.5) else Z_B2 + 3.4, -sg),):
            bx(f"exs_{kk}", (xx - 0.04, yc - 0.32, zz - 0.32), (xx + 0.04, yc + 0.32, zz - 0.08), M["panel_white"], "Stairs", 0.005)
            q = B.image_quad(f"exsf_{kk}", (xx + face * 0.041, yc, zz - 0.2), (0, face, 0) if face > 0 else (0, face, 0), (0, 0, 1), 0.6, 0.22, M["exit_green"])
            put(q, "Stairs")
            bx(f"exsr_{kk}", (xx - 0.008, yc - 0.008, zz - 0.08), (xx + 0.008, yc + 0.008, zz), M["steel"], "Stairs")
    print("details", k)


def enable_gpu():
    """Render on every CUDA/OptiX GPU present (Kaggle T4 x2, etc.); used when UMEDA_GPU is set."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            bpy.context.scene.cycles.device = "GPU"
            print("GPU", kind, [d.name for d in gpus], flush=True)
            return
    print("GPU requested but none found; rendering on CPU", flush=True)


def build_camera(view, args):
    cam = bpy.data.cameras.new("cam")
    cam.sensor_width = 36
    cam.clip_start = 0.05
    cam.clip_end = 2000
    ob = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    if view == "platform":
        cam.lens = 16
        ob.location = (args.cx if args.cx is not None else 40.0, -9.8, Z_B2 + 1.6)
        ob.rotation_euler = (math.radians(90 + 13), 0, math.radians(-76))
        cam.lens = 15
        cam.dof.use_dof = True
        cam.dof.focus_distance = 14
        cam.dof.aperture_fstop = 2.0
    elif view == "arch":
        # matches the 2024 Commons photo taken at the south end of the arch, looking north
        ob.location = (89.0, -1.6, Z_B2 + 1.5)
        ob.rotation_euler = (math.radians(90 + 6), 0, math.radians(105))
        cam.lens = 24
    elif view == "stairtop":
        # matches the 2022 Commons photo from the landing of stair J, looking north into the arch
        ob.location = (95.35, -7.6, Z_B2 + 14 * 5.8 / 34 + 1.55)
        ob.rotation_euler = (math.radians(90 - 21), 0, math.radians(90))
        cam.lens = 16
    elif view == "track1":
        # track-1 side under the flat ceiling, looking north (wp_plat1, 2023)
        ob.location = (82.5, 12.6, Z_B2 + 1.5)
        ob.rotation_euler = (math.radians(90 + 4), 0, math.radians(94))
        cam.lens = 24
    elif view == "concourse":
        # corridor inside the north gate, behind the top of stair H, looking south (2023 photo)
        st = PLAN["stairs"]["H"]
        ob.location = (st["x_top"] - 6.0, -5.2, Z_B1 + 1.6)
        ob.rotation_euler = (math.radians(90 - 2), 0, math.radians(-90))
        cam.lens = 22
        cam.dof.use_dof = True
        cam.dof.focus_distance = 12
        cam.dof.aperture_fstop = 2.4
    elif view == "gate":
        # unpaid side of the north gate, looking south at the gate line (2023 photos)
        ob.location = (-132.6, -4.8, Z_B1 + 1.55)
        ob.rotation_euler = (math.radians(90 + 7), 0, math.radians(-116))
        cam.lens = 24
    elif view == "top":
        # same framing as the map crop x 280..1720, y 390..1250 px (page layout, 1 px = 1 map px)
        cam.type = "ORTHO"
        S = PLAN["scale"]
        cam.ortho_scale = 1440 * S
        cx = (1000 - PLAN["origin_px"][0]) * S
        cy = -(820 - PLAN["origin_px"][1]) * S
        ob.location = (cx, cy, 300)
        ob.rotation_euler = (0, 0, 0)
    elif view == "overview":
        cam.type = "ORTHO"
        cam.ortho_scale = 292
        tgt = Vector((-2.0, 30.0, -2.0))
        d = Vector((-0.12, -0.72, 0.68)).normalized()
        ob.location = tgt + d * 400
        ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()


def top_setup():
    # page layout of the source map: platform drawing 493 px below the concourse drawing
    dy = PLAN["plat_dy"] * PLAN["scale"]
    for cname in ("B2", "Stairs", "Vault", "B2_ceiling", "B2_lights"):
        for ob in coll(cname).objects:
            ob.location.y -= dy
    for cname in ("B1_ceiling", "B2_ceiling", "B1_lights", "B2_lights", "Vault", "Stairs"):
        coll(cname).hide_render = True
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = 1440, 860
    world = sc.world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.5
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(20), math.radians(15), 0)


def overview_setup():
    # exploded view: lift B1 above B2, hide ceilings, light like a model under daylight
    for name in ("B1", "Stairs_top"):
        pass
    # like the source map: concourse drawn beside (east of) the platform level
    for ob in coll("B1").objects:
        ob.location.y += 62.0
        ob.location.z += 6.0
    for cname in ("B1_ceiling", "B2_ceiling", "B1_lights", "B2_lights"):
        coll(cname).hide_render = True
    coll("Vault").hide_render = True
    world = bpy.context.scene.world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.92, 0.93, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.9
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    sun.angle = math.radians(8)
    so = bpy.data.objects.new("sun", sun)
    bpy.context.scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(40), math.radians(10), math.radians(-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="platform", choices=["platform", "arch", "stairtop", "track1", "concourse", "gate", "overview", "top"])
    ap.add_argument("--out", default="../../render/umeda_test.exr")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cam", default="", help="override: x,y,z,pitch_deg,yaw_deg,lens")
    ap.add_argument("--blend", default="")
    ap.add_argument("--multi", default="", help="several camera specs separated by ';'")
    ap.add_argument("--border", default="", help="render only this region: x0,y0,x1,y1 as fractions (y up)")
    args = ap.parse_args()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ns = argparse.Namespace(out=args.out, width=args.width, samples=args.samples, noise=args.noise)
    B.render_settings(ns)
    if os.environ.get("UMEDA_GPU"):
        enable_gpu()
    bpy.context.scene.cycles.blur_glossy = 0.05     # keep reflections crisp
    bpy.context.scene.cycles.sample_clamp_indirect = 12.0
    M = materials()
    print(f"materials {time.time() - t0:.1f}s")
    build_b1(M)
    build_continuation(M)
    build_b2(M)
    build_stairs(M)
    build_gates(M)
    build_details(M)
    UP.build(sys.modules[__name__], M)
    build_exits(M)
    build_perimeter(M)
    if args.view not in ("top", "overview"):
        build_b1_lights(M)
        build_fill_lights(M)
    build_camera(args.view, args)
    if args.cam:
        x, y, z, pitch, yaw, lens = [float(v) for v in args.cam.split(",")]
        cam = bpy.context.scene.camera
        cam.location = (x, y, z)
        cam.rotation_euler = (math.radians(90 + pitch), 0, math.radians(yaw))
        cam.data.lens = lens
    if args.view in ("arch", "stairtop", "track1", "platform"):
        coll("B1_lights").hide_render = True
    if args.view in ("concourse", "gate"):
        coll("B2_lights").hide_render = True
    if args.view == "overview":
        overview_setup()
    if args.view == "top":
        top_setup()
    print(f"built {time.time() - t0:.1f}s objects={len(bpy.data.objects)}")
    if args.border:
        r = bpy.context.scene.render
        r.use_border = True
        r.use_crop_to_border = True
        r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = [float(v) for v in args.border.split(",")]
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
    if args.multi:
        base = args.out[:-4]
        for j, spec in enumerate(args.multi.split(";")):
            x, y, z, pitch, yaw, lens = [float(v) for v in spec.split(",")]
            cam = bpy.context.scene.camera
            cam.location = (x, y, z)
            cam.rotation_euler = (math.radians(90 + pitch), 0, math.radians(yaw))
            cam.data.lens = lens
            bpy.context.scene.render.filepath = os.path.abspath(f"{base}_c{j}.exr")
            bpy.ops.render.render(write_still=True)
        return
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"render done {time.time() - t1:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
