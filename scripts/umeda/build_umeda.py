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
from scipy import ndimage  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
PLAN = json.load(open(os.path.join(ROOT, "cache", "umeda", "plan.json")))

# ------------------------------------------------------------- heights (estimates, metres)
Z_B1 = 0.0          # concourse floor
H_B1 = 3.0          # concourse ceiling height
Z_B2 = -5.8         # platform floor (stairs: 34 risers of 0.17 m)
Z_B2_CEIL = -2.2    # flat platform ceiling (3.6 m clear)
Z_TRACK = Z_B2 - 1.1
Z_SPRING = Z_B2 + 3.0   # arch springing line
Z_CROWN = 1.3           # arch crown (rises into the B1 level, hence the void on the B1 map)
X_MIN, X_MAX = -112.0, 133.0   # tunnel extent modelled beyond the platform ends

COLL = {}


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


def materials():
    M = {}
    M["b1_floor"] = stone("b1_floor2", 0.6, 0.50, 5, rough=0.16, tint=(1.0, 1.0, 0.99))
    # platform: small beige-grey pavers in running bond (photos 2019/2022)
    M["b2_floor"] = panels("b2_pavers", 0.20, 0.10, 0.004, (0.40, 0.37, 0.32), (0.16, 0.15, 0.13), 0.35, bond=0.5, seed=3, var=0.06)
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
    M["steel_brushed"] = B.plain("steel_brushed", (0.72, 0.72, 0.71), 0.32, metal=1.0)
    M["psd_glass"] = B.plain("psd_glass", (0.05, 0.06, 0.07), 0.05, coat=1.0)
    M["gate_body"] = B.plain("gate_body", (0.10, 0.10, 0.11), 0.3, metal=0.5)
    M["gate_side"] = B.plain("gate_side", (0.62, 0.62, 0.62), 0.3, metal=1.0)
    M["gate_top"] = B.plain("gate_top", (0.02, 0.02, 0.025), 0.12, coat=1.0)
    M["gate_blue"] = B.emissive("gate_blue", (0.1, 0.35, 1.0), 6.0)
    M["gate_green"] = B.emissive("gate_green", (0.1, 1.0, 0.25), 5.0)
    M["gate_yellow"] = B.plain("gate_yellow", (0.85, 0.7, 0.05), 0.4)
    M["stair"] = B.plain("stair", (0.40, 0.40, 0.40), 0.45)
    M["nosing"] = B.plain("nosing", (0.75, 0.6, 0.12), 0.5)
    M["glass_rail"] = B.plain("glass_rail", (0.55, 0.6, 0.6), 0.06, coat=1.0)
    M["tactile"] = B.plain("tactile", (0.74, 0.47, 0.03), 0.45)
    M["bench"] = B.plain("bench", (0.55, 0.55, 0.56), 0.3, metal=0.8)
    M["downlight"] = B.emissive("downlight", (1.0, 0.95, 0.88), 40.0)
    M["glassbox"] = B.emissive("glassbox", (1.0, 0.96, 0.86), 6.0)
    M["cove"] = B.emissive("cove", (1.0, 0.93, 0.82), 10.0)
    M["led_line"] = B.emissive("led_line", (0.95, 0.98, 1.0), 25.0)
    gfx = UG.build_all()
    for k in ("name_board", "track1", "track2", "exit_sign", "gate_nw", "gate_ne", "gate_n", "gate_s", "exit_b1", "plat_sign"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.2)
    for k in ("ad0", "ad1", "ad2"):
        M[k] = B.printed(k, gfx[k], rough=0.3, coat=0.0, emission=1.6)
    M["led_vision"] = B.printed("led_vision", gfx["led_vision"], rough=0.4, coat=0.0, emission=4.0)
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
    for i, poly in enumerate(PLAN["b1_ceiling"]):
        prism(f"b1_ceil_{i}", poly, Z_B1 + H_B1, Z_B1 + H_B1 + 0.05, M["ceil"], "B1_ceiling", uv_scale=1.2)
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
    """Inner corridor behind the north gate: shallow barrel vault, linear lights, ads, tactile line."""
    c = "B1"
    cr = PLAN["b1_corridor"]
    x0, x1, yn, ys = cr["x0"], cr["x1"], cr["y_n"], cr["y_s"]
    prof, (ym, zcen, R, a0) = arch_profile(ys, yn, Z_B1 + H_B1 - 0.1, Z_B1 + H_B1 + 1.0, n=32)
    verts, faces = [], []
    for x in (x0, x1):
        verts += [(x, y, z) for (y, z) in prof]
    m = len(prof)
    faces = [(i, i + 1, m + i + 1, m + i) for i in range(m - 1)]
    me = bpy.data.meshes.new("b1_vault")
    me.from_pydata(verts, [], faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            j, i = divmod(vi, m)
            uvl.data[li].uv = ((x0, x1)[j] / 2.4, (R * (2 * a0 * i / (m - 1))) / 2.4)
    me.materials.append(M["vault"])
    coll("B1_ceiling").objects.link(bpy.data.objects.new("b1_vault", me))
    # three continuous light lines along the vault
    for a in (-0.42 * a0, 0.0, 0.42 * a0):
        y = ym + (R - 0.03) * math.sin(a)
        z = zcen + (R - 0.03) * math.cos(a)
        ob = B.box("b1_vl", (0, 0, 0), (x1 - x0, 0.10, 0.03), M["led_line"])
        ob.location = ((x0 + x1) / 2, y, z)
        ob.rotation_euler = (-a, 0, 0)
        put(ob, "B1_lights")
        for xl in np.arange(x0 + 3, x1 - 2, 6.0):
            L = B.area_light(f"b1vA_{xl:.0f}_{a:.2f}", (xl, y, z - 0.05), 5.6, 0.12, 55.0, (1.0, 0.97, 0.93), camera=False)
            L.rotation_euler = (-a, 0, 0)
            put(L, "B1_lights")
    # ads along both corridor walls, tactile guide line along the centre
    for i, x in enumerate(np.arange(x0 + 5, x1 - 3, 5.5)):
        for y, right, nm in ((yn - 0.45, (1, 0, 0), "n"), (ys + 0.25, (-1, 0, 0), "s")):
            B.image_quad(f"b1ad_{nm}_{i}", (x, y, Z_B1 + 1.55), right, (0, 0, 1), 2.6, 1.45, M[f"ad{(i + (nm == 's')) % 3}"])
            bx(f"b1adf_{nm}_{i}", (x - 1.36, y + 0.002 if nm == "n" else y - 0.25, Z_B1 + 0.78), (x + 1.36, y + 0.25 if nm == "n" else y - 0.002, Z_B1 + 2.32), M["steel"], c)
    for ob in list(bpy.context.scene.collection.objects):
        if ob.name.startswith("b1ad_"):
            put(ob, c)
    yt = (yn + ys) / 2 + 1.2
    bx("b1_tactile_corr", (x0 + 1, yt - 0.15, Z_B1), (x1 - 1, yt + 0.15, Z_B1 + 0.005), M["tactile"], c)
    # platform signs hanging from the vault before each stair
    for k in ("F", "G", "H"):
        st = PLAN["stairs"][k]
        xs = st["x_top"] - 3.0
        hang_sign(f"ps_{k}", xs, (yn + ys) / 2, M["plat_sign"], M["exit_b1"], c, z_top=Z_B1 + H_B1 + 0.9)



def hang_sign(name, x, y, mat_front, mat_back, cname, along="x", z_top=None):
    w, h = 2.6, 0.48
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
    for i, poly in enumerate(PLAN["b2_walls"]):
        cen, area = poly_centroid_area(poly)
        in_vault = vx0 < cen[0] < vx1 and yv_w < cen[1] < yv_e
        if in_vault and area < 3.0:
            ztop = vault_z(cen[1])
            prism(f"b2_vcol_{i}", poly, Z_B2 + 0.15, ztop, M["column"], c)
            prism(f"b2_vcolsk_{i}", poly, Z_B2, Z_B2 + 0.15, M["steel_brushed"], c)
            # ad lightbox wraps on the long faces (as in the 2019 photos)
            r = np.array(poly[0])
            (cx0, cy0), (cx1, cy1) = r.min(0), r.max(0)
            for yy, right, nm in ((cy0 - 0.015, (1, 0, 0), "s"), (cy1 + 0.015, (-1, 0, 0), "n")):
                w = (cx1 - cx0) - 0.1
                if w > 0.4:
                    B.image_quad(f"vcad_{i}_{nm}", ((cx0 + cx1) / 2, yy, Z_B2 + 1.65), right, (0, 0, 1), w, 1.7, M[f"ad{i % 3}"])
            for ob in list(bpy.context.scene.collection.objects):
                if ob.name.startswith("vcad_"):
                    put(ob, c)
            continue
        mat = M["column"] if area < 1.5 else M["panel_white"]
        prism(f"b2_wall_{i}", poly, Z_B2, Z_B2_CEIL, mat, c, uv_scale=2.4)
    for i, poly in enumerate(PLAN["b2_ceiling"]):
        prism(f"b2_ceil_{i}", poly, Z_B2_CEIL, Z_B2_CEIL + 0.05, M["ceil"], "B2_ceiling", uv_scale=1.2)

    # track beds, rails, tunnel walls
    ye, yw = PLAN["edges_y"]["up"], PLAN["edges_y"]["lo"]
    te, tw = PLAN["track_wall_y"]["east"], PLAN["track_wall_y"]["west"]
    rect("trackbed_e", X_MIN, ye, X_MAX, te, Z_TRACK, M["ballast"], c)
    rect("trackbed_w", X_MIN, tw, X_MAX, yw, Z_TRACK, M["ballast"], c)
    px0, px1 = PLAN["plat_x"]
    for y, nrm, nm in ((ye, (0, 1, 0), "e"), (yw, (0, -1, 0), "w")):
        vrect(f"plat_edge_{nm}", (px0, y), (px1, y), Z_TRACK, Z_B2, M["concrete"], c, nrm)
    vrect("tunnel_e", (X_MIN, te), (X_MAX, te), Z_TRACK, Z_B2_CEIL, M["wall_tile"], c, (0, -1, 0), 2.4)
    # west wall: up to the arch springing under the vault, else to the flat ceiling
    vrect("tunnel_w_n", (X_MIN, tw), (vx0, tw), Z_TRACK, Z_B2_CEIL, M["wall_tile"], c, (0, 1, 0), 2.4)
    vrect("tunnel_w_v", (vx0, tw), (vx1, tw), Z_TRACK, Z_SPRING, M["wall_tile"], c, (0, 1, 0), 2.4)
    vrect("tunnel_w_s", (vx1, tw), (X_MAX, tw), Z_TRACK, Z_B2_CEIL, M["wall_tile"], c, (0, 1, 0), 2.4)
    rect("ceil_track_e_n", X_MIN, ye, px0, te, Z_B2_CEIL, M["ceil"], "B2_ceiling", False, 1.2)
    rect("ceil_track_w_n", X_MIN, tw, px0, yw, Z_B2_CEIL, M["ceil"], "B2_ceiling", False, 1.2)
    rect("ceil_track_e_s", px1, ye, X_MAX, te, Z_B2_CEIL, M["ceil"], "B2_ceiling", False, 1.2)
    rect("ceil_track_w_s", px1, tw, X_MAX, yw, Z_B2_CEIL, M["ceil"], "B2_ceiling", False, 1.2)
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



def vault_z(y):
    yw, ye = PLAN["track_wall_y"]["west"], PLAN["vault"]["y_east"]
    prof, (ym, zcen, R, a0) = arch_profile(yw, ye, Z_SPRING, Z_CROWN)
    return zcen + math.sqrt(max(R * R - (y - ym) ** 2, 0.0))


def build_vault(M):
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    yw, ye = PLAN["track_wall_y"]["west"], PLAN["vault"]["y_east"]
    prof, (ym, zcen, R, a0) = arch_profile(yw, ye, Z_SPRING, Z_CROWN, n=64)
    m = len(prof)
    verts = [(x, y, z) for x in (vx0, vx1) for (y, z) in prof]
    faces = [(i, i + 1, m + i + 1, m + i) for i in range(m - 1)]
    me = bpy.data.meshes.new("vault")
    me.from_pydata(verts, [], faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            j, i = divmod(vi, m)
            uvl.data[li].uv = ((vx0, vx1)[j] / 2.4, (R * 2 * a0 * i / (m - 1)) / 2.4)
    me.materials.append(M["vault"])
    coll("Vault").objects.link(bpy.data.objects.new("vault", me))
    # end walls: strip between the arch and the flat-ceiling line
    for x, nm in ((vx0, "n"), (vx1, "s")):
        pts, faces = [], []
        for i, (y, z) in enumerate(prof):
            pts += [(x, y, z), (x, y, Z_B2_CEIL)]
            if i:
                a = 2 * (i - 1)
                faces.append((a, a + 2, a + 3, a + 1))
        me = bpy.data.meshes.new("vault_end_" + nm)
        me.from_pydata(pts, [], faces)
        me.update()
        me.materials.append(M["panel_white"])
        coll("Vault").objects.link(bpy.data.objects.new("vault_end_" + nm, me))
    # LED vision: 40 m x 4 m on the half of the arch next to the central wall
    L = 40.0
    xs0 = vx0 + (vx1 - vx0 - L) / 2
    Rl = R - 0.06
    a_from = a0 - 0.5 / Rl
    a_to = a_from - 4.0 / Rl
    nseg = 24
    verts = []
    for x in (xs0, xs0 + L):
        for i in range(nseg + 1):
            a = a_from + (a_to - a_from) * i / nseg
            verts.append((x, ym + Rl * math.sin(a), zcen + Rl * math.cos(a)))
    faces = [(i, nseg + 1 + i, nseg + 2 + i, i + 1) for i in range(nseg)]
    me = bpy.data.meshes.new("led_vision")
    me.from_pydata(verts, [], faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            j, i = divmod(vi, nseg + 1)
            uvl.data[li].uv = (float(j), i / nseg)
    me.materials.append(M["led_vision"])
    coll("Vault").objects.link(bpy.data.objects.new("led_vision", me))
    # LED frame
    for a in (a_from + 0.08 / Rl, a_to - 0.08 / Rl):
        ob = B.box("led_frame", (0, 0, 0), (L + 0.3, 0.12, 0.08), M["panel_grey"])
        ob.location = (xs0 + L / 2, ym + (Rl - 0.02) * math.sin(a), zcen + (Rl - 0.02) * math.cos(a))
        ob.rotation_euler = (-a, 0, 0)
        put(ob, "Vault")
    # rows of small round downlights across the rest of the arch
    k = 0
    for a in np.linspace(-0.88 * a0, a_to - 0.6 / Rl, 7):
        y = ym + (R - 0.015) * math.sin(a)
        z = zcen + (R - 0.015) * math.cos(a)
        for x in np.arange(vx0 + 0.8, vx1 - 0.6, 1.2):
            ob = B.cylinder(f"dl_{k}", (0, 0, 0), 0.055, 0.02, M["downlight"], segs=16)
            ob.location = (x, y, z)
            ob.rotation_euler = (-a, 0, 0)
            put(ob, "Vault")
            k += 1
    # cove light along the west springing
    bx("cove_w", (vx0, yw + 0.12, Z_SPRING - 0.05), (vx1, yw + 0.30, Z_SPRING + 0.02), M["cove"], "Vault")
    # hanging square glass light boxes over the platform
    for x in np.arange(vx0 + 4.0, vx1 - 3.0, 7.0):
        for y in (-3.0, -9.3):
            zb = Z_B2 + 4.3
            bx(f"gb_frame_{x:.0f}_{y:.0f}", (x - 0.75, y - 0.75, zb), (x + 0.75, y + 0.75, zb + 0.45), M["steel"], "Vault", 0.01)
            bx(f"gb_glass_{x:.0f}_{y:.0f}", (x - 0.70, y - 0.70, zb - 0.01), (x + 0.70, y + 0.70, zb + 0.01), M["glassbox"], "Vault")
            zt = vault_z(y)
            for dx, dy in ((-0.6, -0.6), (0.6, 0.6)):
                bx(f"gb_rod_{x:.0f}_{y:.0f}_{dx}", (x + dx - 0.008, y + dy - 0.008, zb + 0.45), (x + dx + 0.008, y + dy + 0.008, zt), M["steel"], "Vault")
            Lb = B.area_light(f"gbA_{x:.0f}_{y:.0f}", (x, y, zb - 0.02), 1.4, 1.4, 260.0, (1.0, 0.95, 0.86), camera=False)
            put(Lb, "B2_lights")
    # fill light from the arch downlights (invisible to the camera)
    for x in np.arange(vx0 + 4, vx1 - 3, 8.0):
        ob = B.area_light(f"vaultA_{x:.0f}", (x, -5.0, Z_CROWN - 0.7), 7.0, 11.0, 700.0, (1.0, 0.95, 0.88), camera=False)
        ob.visible_glossy = False
        put(ob, "B2_lights")
    build_central_wall_face(M)


def build_central_wall_face(M):
    """West face of the central wall under the arch: beige panels, arched recesses, ads, cornice."""
    vx0, vx1 = PLAN["vault"]["x0"], PLAN["vault"]["x1"]
    y = PLAN["vault"]["y_east"] - 0.22     # drawn wall line is dilated by ~0.2 m in plan.py
    ztop = Z_SPRING + 0.4
    vrect("cw_face", (vx0, y), (vx1, y), Z_B2, ztop, M["beige"], "B2", (0, -1, 0), 2.4)
    bx("cw_cornice", (vx0, y - 0.0, ztop - 0.05), (vx1, y + 0.0 - 0.16, ztop + 0.05), M["panel_white"], "B2")
    bx("cw_skirt", (vx0, y - 0.03, Z_B2), (vx1, y, Z_B2 + 0.12), M["steel_brushed"], "B2")
    k = 0
    for x in np.arange(vx0 + 4.5, vx1 - 3, 9.0):
        # arched recess (door) 1.8 m wide, 3.0 m high
        w, hh = 1.8, 2.1
        pts = [(x - w / 2, Z_B2), (x + w / 2, Z_B2), (x + w / 2, Z_B2 + hh)]
        for i in range(1, 16):
            a = math.pi * i / 16
            pts.append((x + w / 2 * math.cos(a), Z_B2 + hh + w / 2 * math.sin(a)))
        pts.append((x - w / 2, Z_B2 + hh))
        for off, mat, nm in ((0.02, M["panel_white"], "door"),):
            verts = [(px_, y - off, pz) for (px_, pz) in pts]
            me = bpy.data.meshes.new(f"arch_{k}")
            me.from_pydata(verts, [], [list(range(len(verts)))])
            me.update()
            box_uv(me)
            me.materials.append(mat)
            coll("B2").objects.link(bpy.data.objects.new(f"arch_{k}", me))
        # arch surround (stainless band)
        for i in range(len(pts) - 1):
            (ax, az), (bx_, bz) = pts[i], pts[i + 1]
            if az == Z_B2 and bz == Z_B2:
                continue
            ln = math.hypot(bx_ - ax, bz - az)
            ob = B.box(f"archf_{k}_{i}", (0, 0, 0), (ln, 0.05, 0.08), M["steel"])
            ob.location = ((ax + bx_) / 2, y - 0.03, (az + bz) / 2)
            ob.rotation_euler = (0, -math.atan2(bz - az, bx_ - ax), 0)
            put(ob, "B2")
        # ad lightbox between arches
        xa = x + 4.5
        if xa < vx1 - 2.5:
            B.image_quad(f"cad_{k}", (xa, y - 0.04, Z_B2 + 1.6), (-1, 0, 0), (0, 0, 1), 2.6, 1.45, M[f"ad{k % 3}"])
            bx(f"cadf_{k}", (xa - 1.36, y - 0.035, Z_B2 + 0.83), (xa + 1.36, y, Z_B2 + 2.37), M["steel"], "B2")
        k += 1
    for ob in list(bpy.context.scene.collection.objects):
        if ob.name.startswith("cad_"):
            put(ob, "B2")


def build_psd(M):
    """Platform screen doors along both edges (one door per car door position, 10 cars)."""
    ye, yw = PLAN["edges_y"]["up"], PLAN["edges_y"]["lo"]
    px0, px1 = PLAN["plat_x"]
    for y, side, nm, length in ((ye, -1, "e", 197.0), (yw, 1, "w", 201.0)):
        y_in = y + side * 0.25
        xa = (px0 + px1) / 2 - length / 2
        car = length / 10
        k = 0
        for cidx in range(10):
            for d in range(4):
                xd = xa + cidx * car + car * (0.14 + d * 0.24)
                # door leaves (closed) 2.0 m opening
                bx(f"psd_{nm}_door_{k}", (xd, y_in - 0.03, Z_B2), (xd + 2.0, y_in + 0.03, Z_B2 + 1.3), M["steel_brushed"], "B2", 0.004)
                bx(f"psd_{nm}_win_{k}", (xd + 0.2, y_in + side * 0.035 - 0.003, Z_B2 + 0.65), (xd + 1.8, y_in + side * 0.035 + 0.003, Z_B2 + 1.15), M["psd_glass"], "B2")
                k += 1
        # continuous fixed panels and top rail
        bx(f"psd_{nm}_base", (xa - 1, y_in - 0.12, Z_B2), (xa + length + 1, y_in + 0.12, Z_B2 + 0.08), M["steel"], "B2")
        bx(f"psd_{nm}_top", (xa - 1, y_in - 0.10, Z_B2 + 1.3), (xa + length + 1, y_in + 0.10, Z_B2 + 1.36), M["steel"], "B2", 0.01)
        bx(f"psd_{nm}_fixed", (xa - 1, y_in - 0.02, Z_B2 + 0.08), (xa + length + 1, y_in + 0.02, Z_B2 + 1.3), M["psd_glass"], "B2")


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
                bx(f"ledl_{k}", (x - 2.2, y - 0.06, Z_B2_CEIL - 0.03), (x + 2.2, y + 0.06, Z_B2_CEIL), M["led_line"], "B2_lights")
                B.area_light(f"ledA_{k}", (x, y, Z_B2_CEIL - 0.035), 4.4, 0.12, 180.0, col, camera=False)
                put(bpy.data.objects[f"ledA_{k}"], "B2_lights")
                k += 1
            x += 5.6
    # fill light under the vault (the discs are mesh lights; add soft area lights for noise-free bounce)
    for x in np.arange(vx0 + 4, vx1 - 3, 8.0):
        ob = B.area_light(f"vaultA_{x:.0f}", (x, -6.7, Z_CROWN - 0.6), 6.0, 10.0, 900.0, (1.0, 0.95, 0.88), camera=False)
        ob.visible_glossy = False
        put(ob, "B2_lights")


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
                B.image_quad(f"nb_{nm}_{k}", (x, y, Z_B2 + 1.9), right, (0, 0, 1), 3.0, 0.7, M["name_board"])
            else:
                B.image_quad(f"ad_{nm}_{k}", (x, y, Z_B2 + 1.8), right, (0, 0, 1), 3.6, 2.0, M[f"ad{k % 3}"])
                bx(f"adf_{nm}_{k}", (x - 1.85, y - 0.02 if nm == "w" else y, Z_B2 + 0.75), (x + 1.85, y if nm == "w" else y + 0.02, Z_B2 + 2.85), M["steel"], "B2")
        for ob in list(bpy.context.scene.collection.objects):
            put(ob, "B2")
        k += 1
        x += 9.0
    # hanging track number signs over the platform
    for x in np.arange(PLAN["plat_x"][0] + 20, PLAN["plat_x"][1] - 15, 30.0):
        for y, img, nm in ((15.0, "track1", "e"), (-4.5, "track2", "w")):
            zc = Z_B2_CEIL - 0.6 if not (vx0 < x < vx1 and y < 1) else Z_SPRING + 0.3
            ztop = Z_B2_CEIL if not (vx0 < x < vx1 and y < 1) else Z_CROWN - 0.3
            B.image_quad(f"ts_{nm}_{x:.0f}", (x - 0.06, y, zc), (0, -1, 0), (0, 0, 1), 2.6, 0.52, M[img])
            B.image_quad(f"tsb_{nm}_{x:.0f}", (x + 0.06, y, zc), (0, 1, 0), (0, 0, 1), 2.6, 0.52, M["exit_sign"])
            bx(f"tsbody_{nm}_{x:.0f}", (x - 0.055, y - 1.32, zc - 0.28), (x + 0.055, y + 1.32, zc + 0.28), M["panel_grey"], "B2")
            for dy in (-1.0, 1.0):
                bx(f"tsrod_{nm}_{x:.0f}_{dy}", (x - 0.01, y + dy - 0.01, zc + 0.28), (x + 0.01, y + dy + 0.01, ztop), M["steel"], "B2")
        for ob in list(bpy.context.scene.collection.objects):
            put(ob, "B2")


# ------------------------------------------------------------- stairs, gates, exits

def flight(name, x_top, x_bot, y0, y1, z_top, z_bot, M, cname, landing=True):
    rise = z_top - z_bot
    n = int(round(rise / 0.17))
    riser = rise / n
    run_total = abs(x_bot - x_top)
    sgn = 1 if x_bot > x_top else -1
    tread = 0.30
    land = max(0.0, run_total - n * tread) if landing else 0.0
    half = n // 2
    x = x_top
    z = z_top
    for i in range(n):
        if i == half and land > 0:
            xa, xb = sorted((x, x + sgn * land))
            bx(f"{name}_land", (xa, y0, z_bot), (xb, y1, z), M["stair"], cname)
            x += sgn * land
        z_step = z - riser
        xa, xb = sorted((x, x + sgn * tread))
        bx(f"{name}_s{i}", (xa, y0, z_bot), (xb, y1, z_step), M["stair"], cname)
        xn = x if sgn > 0 else x - 0.04
        bx(f"{name}_n{i}", (xn, y0 + 0.05, z_step - 0.004), (xn + 0.04, y1 - 0.05, z_step + 0.002), M["nosing"], cname)
        x += sgn * tread
        z = z_step
    # handrails (sides + centre)
    for yy in (y0 + 0.08, (y0 + y1) / 2, y1 - 0.08):
        p0 = Vector((x_top, yy, z_top + 0.85))
        p1 = Vector((x_bot, yy, z_bot + 0.85))
        d = p1 - p0
        ob = B.cylinder(f"{name}_hr{yy:.1f}", ((p0.x + p1.x) / 2, yy, (p0.z + p1.z) / 2), 0.022, d.length, M["steel"], axis="X")
        ang = math.atan2(-(p1.z - p0.z), abs(p1.x - p0.x)) * (1 if sgn > 0 else -1)
        for v in ob.data.vertices:
            v.co -= Vector(((p0.x + p1.x) / 2, yy, (p0.z + p1.z) / 2))
        ob.location = ((p0.x + p1.x) / 2, yy, (p0.z + p1.z) / 2)
        ob.rotation_euler = (0, ang, 0)
        put(ob, cname)


def build_stairs(M):
    for k, st in PLAN["stairs"].items():
        y0, y1 = st["y0"], st["y1"]
        flight(f"st{k}", st["x_top"], st["x_bot"], y0, y1, Z_B1, Z_B2, M, "Stairs")
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        # stairwell side walls and the head wall over the stair foot
        for yy, nm in ((y0 - 0.15, "a"), (y1, "b")):
            bx(f"st{k}_wall{nm}", (xa, yy, Z_B2), (xb, yy + 0.15, Z_B1), M["wall_tile"], "Stairs")
        xh = st["x_bot"]
        ha, hb = sorted((xh, xh + (0.3 if st["x_bot"] > st["x_top"] else -0.3)))
        bx(f"st{k}_head", (ha, y0, Z_B2_CEIL), (hb, y1, Z_B1 - 0.35), M["panel_white"], "Stairs")
        # glass balustrade around the B1 opening (not on the entry side)
        for yy in (y0, y1):
            bx(f"st{k}_bal{yy:.1f}", (xa, yy - 0.02, Z_B1), (xb, yy + 0.02, Z_B1 + 1.1), M["glass_rail"], "B1")
            bx(f"st{k}_balt{yy:.1f}", (xa, yy - 0.04, Z_B1 + 1.1), (xb, yy + 0.04, Z_B1 + 1.14), M["steel"], "B1")
        xe = st["x_bot"]
        bx(f"st{k}_bale", (xe - 0.02, y0, Z_B1), (xe + 0.02, y1, Z_B1 + 1.1), M["glass_rail"], "B1")



def build_gates(M):
    """Automatic ticket gates: dark stainless cabinets with reader panels and coloured indicators."""
    k = 0
    for name, pl in PLAN["gates"].items():
        pts = [Vector((p[0], p[1], 0)) for p in pl]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            L = d.length
            t = d.normalized()
            nrm = Vector((-t.y, t.x, 0))
            n = max(1, int(L / 0.95))
            for i in range(n + 1):
                p = a + t * (L * i / n)
                ang = math.atan2(nrm.y, nrm.x)
                parts = [
                    ((0, 0, 0.5), (1.6, 0.22, 1.0), M["gate_body"]),
                    ((0, 0.112, 0.5), (1.5, 0.004, 0.9), M["gate_side"]),
                    ((0, -0.112, 0.5), (1.5, 0.004, 0.9), M["gate_side"]),
                    ((0, 0, 1.015), (1.5, 0.2, 0.03), M["gate_top"]),
                    ((0.55, 0, 1.04), (0.22, 0.17, 0.02), M["gate_blue"]),
                    ((-0.55, 0, 1.04), (0.22, 0.17, 0.02), M["gate_blue"]),
                    ((0.79, 0, 0.75), (0.02, 0.15, 0.22), M["gate_green"]),
                    ((-0.79, 0, 0.75), (0.02, 0.15, 0.22), M["gate_green"]),
                    ((0.2, 0, 1.032), (0.5, 0.05, 0.01), M["gate_yellow"]),
                ]
                for j, (off, size, mat) in enumerate(parts):
                    ob = B.box(f"gate_{k}_{j}", (0, 0, 0), size, mat, 0.006 if j == 0 else 0.0)
                    ca, sa = math.cos(ang), math.sin(ang)
                    ob.location = (p.x + ca * off[0] - sa * off[1], p.y + sa * off[0] + ca * off[1], Z_B1 + off[2])
                    ob.rotation_euler = (0, 0, ang)
                    put(ob, "B1")
                k += 1


def build_exits(M):
    for k, e in PLAN["exits"].items():
        c = Vector((e["c"][0], e["c"][1], 0))
        u = Vector((e["up"][0], e["up"][1], 0))
        run, w = e["run"], e["w"]
        rise = min(5.5, run * 0.55)
        n = max(4, int(rise / 0.17))
        ang = math.atan2(u.y, u.x)
        for i in range(n):
            p = c + u * (-run / 2 + (i + 0.5) * run / n)
            h = (i + 1) * rise / n
            ob = B.box(f"ex{k}_{i}", (0, 0, 0), (run / n, w, h), M["stair"])
            for v in ob.data.vertices:
                v.co.z += h / 2
            ob.location = (p.x, p.y, Z_B1)
            ob.rotation_euler = (0, 0, ang)
            put(ob, "B1")


def build_b1_lights(M):
    """Square ceiling panels on a 3.6 m grid over the walkable concourse."""
    import cv2
    walk = [np.array(poly[0], np.float32).reshape(-1, 1, 2) for poly in PLAN["b1_walk"]]
    xs = np.arange(-135, 135, 4.8)
    ys = np.arange(-50, 50, 4.8)
    k = 0
    for x in xs:
        for y in ys:
            if any(cv2.pointPolygonTest(r, (float(x), float(y)), False) >= 0 for r in walk):
                bx(f"b1l_{k}", (x - 0.3, y - 0.3, H_B1 - 0.01), (x + 0.3, y + 0.3, H_B1), M["led_line"], "B1_lights")
                ob = B.area_light(f"b1A_{k}", (x, y, H_B1 - 0.015), 0.6, 0.6, 110.0, (1.0, 0.97, 0.93), camera=False)
                put(ob, "B1_lights")
                k += 1
    print("B1 lights", k)


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
    elif view == "concourse":
        cam.lens = 18
        ob.location = (-85.0, -6.0, Z_B1 + 1.6)
        ob.rotation_euler = (math.radians(90 + 9), 0, math.radians(-90))
        cam.lens = 15
        cam.dof.use_dof = True
        cam.dof.focus_distance = 12
        cam.dof.aperture_fstop = 2.4
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
    ap.add_argument("--view", default="platform", choices=["platform", "concourse", "overview", "top"])
    ap.add_argument("--out", default="../../render/umeda_test.exr")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cam", default="", help="override: x,y,z,pitch_deg,yaw_deg,lens")
    ap.add_argument("--blend", default="")
    ap.add_argument("--multi", default="", help="several camera specs separated by ';'")
    args = ap.parse_args()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ns = argparse.Namespace(out=args.out, width=args.width, samples=args.samples, noise=args.noise)
    B.render_settings(ns)
    bpy.context.scene.cycles.blur_glossy = 0.6
    M = materials()
    print(f"materials {time.time() - t0:.1f}s")
    build_b1(M)
    build_b2(M)
    build_stairs(M)
    build_gates(M)
    build_exits(M)
    if args.view == "concourse":
        build_b1_lights(M)
    build_camera(args.view, args)
    if args.cam:
        x, y, z, pitch, yaw, lens = [float(v) for v in args.cam.split(",")]
        cam = bpy.context.scene.camera
        cam.location = (x, y, z)
        cam.rotation_euler = (math.radians(90 + pitch), 0, math.radians(yaw))
        cam.data.lens = lens
    if args.view == "overview":
        overview_setup()
    if args.view == "top":
        top_setup()
    print(f"built {time.time() - t0:.1f}s objects={len(bpy.data.objects)}")
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
