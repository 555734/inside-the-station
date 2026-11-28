"""Module kit of the underground maze: a fictional Higaoka Metro station under ひかり地下街, an
underground shopping street that runs on for blocks (Blender, run with bpy).

The maze is laid out on a grid of 16 m cells. Every module fills one or more cells and meets its
neighbours at ports: openings 6 m wide and 3 m high in the middle of a cell edge, where every
module has the same section (floor, walls, ceiling, the tactile guide line along the middle, the
ceiling lights on the same 4 m grid), so modules join without a seam in any order and turned any
way. Floors, walls and ceilings of a module are made from its floor plan (a union of rectangles):
walls stand exactly on the outline and are left out only at its ports, so there is no stray wall,
gap or floating part. Each module is lit (baked) once and used as often as the layout wants.

  python3 -c "import kit; kit.build_all()"        (inside Blender's Python, see export_dungeon.py)
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "umeda"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
os.environ.setdefault("UMEDA_EXPORT", "1")      # plain materials the app can show (no procedural layers)
import build_scene as B          # noqa: E402
import build_umeda as BU         # noqa: E402  (materials, gates, fences)
import umeda_props as UP         # noqa: E402
import dgfx                      # noqa: E402
import graphics as GX            # noqa: E402
from shapely.geometry import Point, box as sbox  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

UP.H = BU

C = 16.0          # cell size (m)
HC = C / 2
HW = 3.0          # half width of a corridor
CH = 3.0          # corridor ceiling height
UV_FLOOR = C / 7  # floor / wall texture repeat that divides the cell: the pattern runs on across joints
UV_CEIL = C / 13
LIGHT_W = 120.0   # one ceiling light (W): lights on a 4 m grid at x, y = 2 (mod 4)
Z2 = -7.0         # platform level

MODULES = {}      # name -> {"cells": [(i, j)], "ports": [(x, y, side)], "level", "build"}
CUR = None        # collection of the module being built
M = {}


# ------------------------------------------------------------------ small helpers

def put(ob, deco=False, col=True):
    BU.put(ob, CUR)
    if deco or not col:
        ob["nocol"] = 1
    return ob


def bx(name, lo, hi, mat, deco=False, bevel=0.0):
    return put(B.box_mm(f"{CUR}_{name}", lo, hi, mat, bevel), deco)


def quad(name, c, right, up, w, h, mat, deco=True):
    """An image facing right x up (a picture read with 'right' to the viewer's right): on a wall
    whose room side is +y that is right = (-1, 0, 0)."""
    return put(B.image_quad(f"{CUR}_{name}", c, right, up, w, h, mat), deco)


class Frame:
    """Objects made inside are moved by a rotation about z (deg) and a translation:
    with Frame(origin, angle): build in local coordinates."""
    def __init__(self, origin, angle):
        self.T = Matrix.Translation(Vector((origin[0], origin[1], origin[2] if len(origin) > 2 else 0.0))) @ \
            Matrix.Rotation(math.radians(angle), 4, "Z")

    def __enter__(self):
        self.before = set(bpy.data.objects)
        return self

    def __exit__(self, *a):
        for o in bpy.data.objects:
            if o not in self.before:
                o.matrix_world = self.T @ o.matrix_world


def wall_frame(a, n):
    """Frame of a wall at point a whose walkable side is n (unit, axis aligned): local x along the
    wall, local y out of it into the room."""
    ang = math.degrees(math.atan2(-n[0], n[1]))
    return Frame((a[0], a[1], a[2] if len(a) > 2 else 0.0), ang)


def geoms(poly):
    return list(poly.geoms) if hasattr(poly, "geoms") else [poly]


def grid_quads(poly):
    """The floor plan cut into rectangles along every x and y of its outline (no T-junctions)."""
    xs, ys = set(), set()
    for g in geoms(poly):
        for ring in [g.exterior] + list(g.interiors):
            for x, y in ring.coords:
                xs.add(round(x, 4))
                ys.add(round(y, 4))
    xs, ys = sorted(xs), sorted(ys)
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if poly.contains(Point((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)):
                out.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return out


def flat(name, poly, z, mat, up=True, uvs=UV_FLOOR, deco=False):
    vid, verts, faces = {}, [], []

    def v(x, y):
        k = (round(x, 4), round(y, 4))
        if k not in vid:
            vid[k] = len(verts)
            verts.append((x, y, z))
        return vid[k]
    for (x0, y0, x1, y1) in grid_quads(poly):
        f = [v(x0, y0), v(x1, y0), v(x1, y1), v(x0, y1)]
        faces.append(f if up else f[::-1])
    ob = B.mesh_obj(f"{CUR}_{name}", verts, faces, mat, {"UVMap": lambda co: (co[0] / uvs, co[1] / uvs)})
    return put(ob, deco)


def outline(poly, openings=()):
    """Wall runs along the outline with the walkable side n: [(a, b, n)], left out where an
    opening box (x0, y0, x1, y1) covers them."""
    from shapely.geometry import LineString
    holes = unary_union([sbox(*o) for o in openings]) if openings else None
    out = []
    for g in geoms(poly):
        g = orient(g, 1.0)
        for ring in [g.exterior] + list(g.interiors):
            cs = list(ring.coords)
            for a, b in zip(cs[:-1], cs[1:]):
                d = (b[0] - a[0], b[1] - a[1])
                L = math.hypot(*d)
                if L < 1e-6:
                    continue
                t = (d[0] / L, d[1] / L)
                n = (-t[1], t[0])
                ls = LineString([a, b])
                parts = [ls] if holes is None else geoms(ls.difference(holes))
                for p in parts:
                    if p.is_empty or p.length < 0.02:
                        continue
                    c = list(p.coords)
                    pa, pb = c[0], c[-1]
                    if (pb[0] - pa[0]) * t[0] + (pb[1] - pa[1]) * t[1] < 0:
                        pa, pb = pb, pa
                    out.append((pa, pb, n))
    return out


def walls(name, runs, z0, z1, mat, skirting=True, uvs=UV_FLOOR):
    for k, (a, b, n) in enumerate(runs):
        put(BU.vrect(f"{CUR}_{name}{k}", a, b, z0, z1, mat, CUR, (n[0], n[1], 0), uvs))
        if skirting:
            lo = (min(a[0], b[0]) + (n[0] * 0.012 if n[0] < 0 else 0), min(a[1], b[1]) + (n[1] * 0.012 if n[1] < 0 else 0), z0)
            hi = (max(a[0], b[0]) + (n[0] * 0.012 if n[0] > 0 else 0), max(a[1], b[1]) + (n[1] * 0.012 if n[1] > 0 else 0), z0 + 0.09)
            bx(f"{name}sk{k}", lo, hi, M["steel_brushed"], deco=True)


def ceiling_lights(poly, z, margin=0.9, power=LIGHT_W, warm=False, name="L"):
    """Square recessed panels on the 4 m grid (x, y = 2 mod 4) inside the plan."""
    inner = poly.buffer(-margin, join_style=2)
    xs = [x for x in np.arange(-64, 64.1, 4.0) + 2.0]
    k = 0
    b = poly.bounds
    for x in xs:
        if not b[0] < x < b[2]:
            continue
        for y in xs:
            if not b[1] < y < b[3] or not inner.contains(Point(x, y)):
                continue
            light_panel(f"{name}{k}", (x, y, z), power, warm)
            k += 1
    return k


def light_panel(name, p, power=LIGHT_W, warm=False, size=(0.6, 0.6)):
    x, y, z = p
    bx(f"{name}f", (x - size[0] / 2 - 0.04, y - size[1] / 2 - 0.04, z - 0.012), (x + size[0] / 2 + 0.04, y + size[1] / 2 + 0.04, z), M["steel"], deco=True)
    quad(f"{name}p", (x, y, z - 0.014), (1, 0, 0), (0, -1, 0), size[0], size[1], M["glassbox_w" if warm else "glassbox"])
    ob = B.area_light(f"{CUR}_{name}A", (x, y, z - 0.03), size[0], size[1], power,
                      (1.0, 0.9, 0.78) if warm else (1.0, 0.97, 0.93), camera=False)
    ob.data.spread = math.radians(160)
    put(ob, deco=True)


def tactile_line(name, a, b, w=0.3):
    """Guide line of bar blocks from a to b (axis aligned), bars along the way."""
    (ax, ay), (bx_, by) = a, b
    L = math.hypot(bx_ - ax, by - ay)
    if L < 0.05:
        return
    t = ((bx_ - ax) / L, (by - ay) / L)
    n = (-t[1], t[0])
    pts = [(ax + n[0] * w / 2, ay + n[1] * w / 2, 0.004), (bx_ + n[0] * w / 2, by + n[1] * w / 2, 0.004),
           (bx_ - n[0] * w / 2, by - n[1] * w / 2, 0.004), (ax - n[0] * w / 2, ay - n[1] * w / 2, 0.004)]
    # loops in the order 3, 2, 1, 0 (the face looks up): u along the line, v across
    uv = [(0, 0), (L / 0.6, 0), (L / 0.6, w / 0.6), (0, w / 0.6)]
    ob = B.mesh_obj(f"{CUR}_{name}", pts, [(3, 2, 1, 0)], M["tactile_bar"], {"UVMap": uv})
    put(ob, deco=True)


def tactile_dots(name, c, s=0.6):
    x, y = c
    put(BU.rect(f"{CUR}_{name}", x - s / 2, y - s / 2, x + s / 2, y + s / 2, 0.0045, M["tactile"], CUR, True, 0.6), deco=True)


def module(name, cells, ports, level="B1"):
    """Decorator: registers a module builder. cells: centres (x, y) of the cells it fills, in its own
    coordinates (one cell: [(0, 0)]); ports: (x, y, side) at the middle of an outer cell edge."""
    def deco(fn):
        MODULES[name] = {"cells": cells, "ports": ports, "level": level, "build": fn}
        return fn
    return deco


def begin(name):
    global CUR
    CUR = "M_" + name
    BU.coll(CUR)


# ------------------------------------------------------------------ wall fixtures (wall at y=0, room at +y)

def poster(name, x, mat_key, w=1.03, h=1.456, zc=1.55):
    bx(f"{name}fr", (x - w / 2 - 0.05, 0.0, zc - h / 2 - 0.05), (x + w / 2 + 0.05, 0.06, zc + h / 2 + 0.05), M["steel_brushed"], deco=True, bevel=0.006)
    quad(f"{name}im", (x, 0.062, zc), (-1, 0, 0), (0, 0, 1), w, h, M[mat_key])


def band_ad(name, x, mat_key, w=2.7, h=1.5, zc=1.6):
    bx(f"{name}fr", (x - w / 2 - 0.06, 0.0, zc - h / 2 - 0.06), (x + w / 2 + 0.06, 0.08, zc + h / 2 + 0.06), M["clad_dark"], deco=True, bevel=0.008)
    quad(f"{name}im", (x, 0.082, zc), (-1, 0, 0), (0, 0, 1), w, h, M[mat_key])


def staff_door(name, x):
    bx(f"{name}fr", (x - 0.55, 0.0, 0.0), (x + 0.55, 0.05, 2.25), M["steel_brushed"], deco=True)
    bx(f"{name}dr", (x - 0.47, 0.0, 0.0), (x + 0.47, 0.06, 2.15), M["door_grey"], deco=True)
    bx(f"{name}hd", (x + 0.30, 0.06, 0.98), (x + 0.40, 0.10, 1.02), M["steel"], deco=True)
    quad(f"{name}pl", (x, 0.062, 1.55), (-1, 0, 0), (0, 0, 1), 0.42, 0.13, M["plate_staff"])


def vending(name, x, drinks="shelf_drinks"):
    """Drink vending machine against the wall (1.0 x 0.75 x 1.83 m)."""
    bx(f"{name}b", (x - 0.5, 0.02, 0.0), (x + 0.5, 0.77, 1.83), M["vend_body"], bevel=0.01)
    quad(f"{name}w", (x - 0.05, 0.775, 1.28), (-1, 0, 0), (0, 0, 1), 0.82, 0.62, M["vend_front"])
    bx(f"{name}t", (x - 0.5, 0.75, 1.62), (x + 0.5, 0.79, 1.80), M["vend_band"], deco=True)
    bx(f"{name}s", (x - 0.42, 0.75, 0.15), (x + 0.20, 0.80, 0.42), M["panel_grey"], deco=True)
    bx(f"{name}c", (x + 0.30, 0.76, 0.80), (x + 0.42, 0.79, 1.00), M["steel"], deco=True)


def bench(name, x, w=1.8):
    bx(f"{name}s", (x - w / 2, 0.15, 0.40), (x + w / 2, 0.60, 0.45), M["bench_wood"], bevel=0.01)
    for sx in (-w / 2 + 0.15, w / 2 - 0.15):
        bx(f"{name}l{sx:.0f}", (x + sx - 0.03, 0.2, 0.0), (x + sx + 0.03, 0.55, 0.40), M["steel_brushed"])


def lockers(name, x, n=4):
    w = 0.42 * n
    bx(f"{name}b", (x - w / 2, 0.02, 0.0), (x + w / 2, 0.57, 1.85), M["locker"], bevel=0.004)
    for i in range(n):
        for j in range(4):
            cx, cz = x - w / 2 + 0.21 + i * 0.42, 0.25 + j * 0.44
            bx(f"{name}d{i}{j}", (cx - 0.19, 0.57, cz - 0.2), (cx + 0.19, 0.585, cz + 0.2), M["locker_door"], deco=True)


def wall_fixtures(runs, plan):
    """Fixtures along the long wall runs of a corridor: plan = {run index: [(kind, s, arg)]}, s the
    distance along the run from its start."""
    for k, items in plan.items():
        a, b, n = runs[k]
        with wall_frame(a, n):
            for j, (kind, s, arg) in enumerate(items):
                nm = f"w{k}_{j}"
                if kind == "poster":
                    poster(nm, s, arg)
                elif kind == "band":
                    band_ad(nm, s, arg)
                elif kind == "ext":
                    UP.extinguisher(f"{CUR}_{nm}", s, 0.0, 1, M, CUR, 0.0)
                elif kind == "hyd":
                    UP.hydrant(f"{CUR}_{nm}", s, 0.0, 1, M, CUR, 0.0)
                elif kind == "aed":
                    UP.aed(f"{CUR}_{nm}", s, 0.0, 1, M, CUR, 0.0)
                elif kind == "door":
                    staff_door(nm, s)
                elif kind == "vend":
                    vending(nm, s)
                elif kind == "bench":
                    bench(nm, s)
                elif kind == "lockers":
                    lockers(nm, s, arg or 4)


def ceiling_dressing(points, z=None):
    """Smoke detectors and camera domes at the given (x, y) on a ceiling at z (default: corridor)."""
    z = CH if z is None else z
    for k, (kind, x, y) in enumerate(points):
        if kind == "smoke":
            UP.smoke_detector(f"{CUR}_smk{k}", x, y, z, M, CUR)
        elif kind == "cam":
            put(B.sphere(f"{CUR}_cam{k}", (x, y, z - 0.07), 0.07, M["cctv"], scale=(1, 1, 0.8)), deco=True)
            put(B.cylinder(f"{CUR}_camb{k}", (x, y, z - 0.01), 0.09, 0.02, M["kiosk_white"], segs=20), deco=True)


def hanging_sign(name, c, facing, mat, w=2.4, h=0.4, ceil=None):
    """Double-sided sign hung from the ceiling (at ceil, default the corridor's) on two rods;
    facing = direction its front looks."""
    CH = globals()["CH"] if ceil is None else ceil
    x, y, z = c
    f = Vector((facing[0], facing[1], 0)).normalized()
    r = Vector((-f.y, f.x, 0))
    put(B.box(f"{CUR}_{name}b", (x, y, z), (abs(r.x) * w + abs(f.x) * 0.08, abs(r.y) * w + abs(f.y) * 0.08, h + 0.04), M["clad_dark"]), deco=True)
    for s in (-1, 1):
        put(B.image_quad(f"{CUR}_{name}f{s}", (x + f.x * 0.041 * s, y + f.y * 0.041 * s, z), tuple(r * s), (0, 0, 1), w - 0.04, h - 0.02, mat), deco=True)
    for s in (-0.35, 0.35):
        p = Vector((x, y, 0)) + r * (w * s)
        put(B.cylinder(f"{CUR}_{name}r{s}", (p.x, p.y, (z + h / 2 + CH) / 2), 0.012, CH - (z + h / 2), M["steel"], segs=8), deco=True)


# ------------------------------------------------------------------ corridor family

def corridor_shell(plan_rects, ports_boxes, cl_paths, junctions=(), h=CH):
    """Floor, ceiling, walls, skirting, guide line and lights of a corridor-type module."""
    poly = unary_union([sbox(*r) for r in plan_rects])
    flat("floor", poly, 0.0, M["dg_floor"])
    flat("ceil", poly, h, M["dg_ceil"], up=False, uvs=UV_CEIL)
    runs = outline(poly, ports_boxes)
    walls("wall", runs, 0.0, h, M["dg_wall"])
    for k, (a, b) in enumerate(cl_paths):
        tactile_line(f"tl{k}", a, b)
    for k, c in enumerate(junctions):
        tactile_dots(f"td{k}", c)
    ceiling_lights(poly, h)
    return poly, runs


# opening boxes of the ports: a hair narrower than the corridor, so the side walls run right up to
# the cell edge (a box touching them would cut them short and leave a slit at every joint)
E_ = 0.005
PORT = {"N": (-HW + E_, HC - 0.05, HW - E_, HC + 0.05), "S": (-HW + E_, -HC - 0.05, HW - E_, -HC + 0.05),
        "E": (HC - 0.05, -HW + E_, HC + 0.05, HW - E_), "W": (-HC - 0.05, -HW + E_, -HC + 0.05, HW - E_)}
PORT_PT = {"N": (0.0, HC), "S": (0.0, -HC), "E": (HC, 0.0), "W": (-HC, 0.0)}


def _runs_by(runs, n, near):
    """Index of the wall run with walkable side n passing nearest the point near."""
    best, bi = 1e9, None
    for i, (a, b, nn) in enumerate(runs):
        if abs(nn[0] - n[0]) > 0.1 or abs(nn[1] - n[1]) > 0.1:
            continue
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d = math.hypot(mid[0] - near[0], mid[1] - near[1])
        if d < best:
            best, bi = d, i
    return bi


@module("I", [(0, 0)], [(0, HC, "N"), (0, -HC, "S")])
def build_I():
    """Straight corridor: lit posters on both walls, extinguisher and hydrant, a camera."""
    poly, runs = corridor_shell([(-HW, -HC, HW, HC)], [PORT["N"], PORT["S"]], [((0, -HC), (0, HC))])
    e = _runs_by(runs, (-1, 0), (HW, 0))     # east wall (walkable side to the west)
    w = _runs_by(runs, (1, 0), (-HW, 0))
    # run of the east wall goes south->north? walls run with the room on their left
    wall_fixtures(runs, {e: [("poster", 2.0, "adp0"), ("ext", 5.6, None), ("poster", 10.0, "poster_travel"), ("poster", 14.0, "adp1")],
                         w: [("poster", 2.0, "poster_museum"), ("hyd", 7.0, None), ("poster", 12.0, "adp2"), ("poster", 14.2, "poster_manners")]})
    ceiling_dressing([("smoke", 0.0, -4.0), ("cam", 1.2, 4.0), ("smoke", 0.0, 4.0)])


@module("I2", [(0, 0)], [(0, HC, "N"), (0, -HC, "S")])
def build_I2():
    """Straight corridor with a rest corner: vending machines, a bench, coin lockers, staff door."""
    poly, runs = corridor_shell([(-HW, -HC, HW, HC)], [PORT["N"], PORT["S"]], [((0, -HC), (0, HC))])
    e = _runs_by(runs, (-1, 0), (HW, 0))
    w = _runs_by(runs, (1, 0), (-HW, 0))
    wall_fixtures(runs, {e: [("vend", 2.2, None), ("vend", 3.3, None), ("bench", 6.0, None), ("lockers", 10.4, 5), ("poster", 14.2, "adp1")],
                         w: [("band", 3.5, "ad0"), ("door", 7.6, None), ("aed", 9.2, None), ("band", 12.6, "ad2")]})
    ceiling_dressing([("smoke", 0.0, -2.0), ("cam", -1.2, 6.0)])


@module("L", [(0, 0)], [(0, -HC, "S"), (HC, 0, "E")])
def build_L():
    """Corner: the corridor turns from south to east (the outer corner with a poster pair)."""
    poly, runs = corridor_shell([(-HW, -HC, HW, HW), (-HW, -HW, HC, HW)], [PORT["S"], PORT["E"]],
                                [((0, -HC), (0, 0)), ((0, 0), (HC, 0))], [(0, 0)])
    n = _runs_by(runs, (0, -1), (0, HW))      # north wall (outer)
    w = _runs_by(runs, (1, 0), (-HW, 0))      # west wall (outer)
    wall_fixtures(runs, {n: [("poster", 2.0, "adp2"), ("poster", 6.0, "poster_ramen")],
                         w: [("ext", 4.0, None), ("poster", 7.6, "adp0")]})
    ceiling_dressing([("smoke", -1.0, 1.0), ("cam", -2.2, 2.2)])


@module("T", [(0, 0)], [(0, HC, "N"), (0, -HC, "S"), (HC, 0, "E")])
def build_T():
    poly, runs = corridor_shell([(-HW, -HC, HW, HC), (0, -HW, HC, HW)], [PORT["N"], PORT["S"], PORT["E"]],
                                [((0, -HC), (0, HC)), ((0, 0), (HC, 0))], [(0, 0)])
    w = _runs_by(runs, (1, 0), (-HW, 0))
    wall_fixtures(runs, {w: [("poster", 2.0, "adp1"), ("band", 8.0, "ad1"), ("hyd", 13.5, None)]})
    hanging_sign("hs", (0.0, -4.5, 2.45), (0, -1), M["mall_sign"])
    ceiling_dressing([("smoke", -1.0, 0.0), ("cam", 2.0, -2.0)])


@module("X", [(0, 0)], [(0, HC, "N"), (0, -HC, "S"), (HC, 0, "E"), (-HC, 0, "W")])
def build_X():
    poly, runs = corridor_shell([(-HW, -HC, HW, HC), (-HC, -HW, HC, HW)], [PORT["N"], PORT["S"], PORT["E"], PORT["W"]],
                                [((0, -HC), (0, HC)), ((-HC, 0), (HC, 0))], [(0, 0)])
    # the four inner corners: a poster on each wall end facing along the corridors
    for k, (cx, cy) in enumerate(((HW, HW), (-HW, HW), (-HW, -HW), (HW, -HW))):
        sx, sy = (1 if cx > 0 else -1), (1 if cy > 0 else -1)
        i = _runs_by(runs, (-sx, 0), (cx, cy + sy * 2.5))
        a, b, nn = runs[i]
        s = 2.5 if math.hypot(a[0] - cx, a[1] - cy) < 0.1 else math.hypot(b[0] - a[0], b[1] - a[1]) - 2.5
        wall_fixtures(runs, {i: [("poster", s, f"adp{k % 3}")]})
    hanging_sign("hs", (0.0, 4.6, 2.45), (0, -1), M["mall_sign"])
    ceiling_dressing([("cam", 2.0, 2.0), ("smoke", -2.0, -2.0)])


@module("EXIT", [(0, 0)], [(0, -HC, "S")])
def build_exit():
    """Dead end of the street: an exit stair up to the street (daylight through the doors at the
    top, which stay shut: the maze is underground)."""
    rise, steps = 0.16, 30
    run = 0.30
    y0 = -3.5                                    # foot of the stair
    y1 = y0 + steps * run                       # top step edge (5.5)
    zt = rise * steps                           # 4.8
    sw = 2.2                                    # half width of the stair hall
    # corridor part to the foot of the stair, then the stair hall (narrower, walls on both sides)
    poly = unary_union([sbox(-HW, -HC, HW, y0), sbox(-sw, y0 - 0.01, sw, HC - 0.3)])
    flat("floor", sbox(-HW, -HC, HW, y0), 0.0, M["dg_floor"])
    flat("ceil", sbox(-HW, -HC, HW, y0), CH, M["dg_ceil"], up=False, uvs=UV_CEIL)
    runs = outline(sbox(-HW, -HC, HW, y0), [PORT["S"], (-sw + E_, y0 - 0.05, sw - E_, y0 + 0.05)])
    walls("wall", runs, 0.0, CH, M["dg_wall"])
    tactile_line("tl", (0, -HC), (0, y0 - 0.6))
    tactile_dots("td", (0, y0 - 0.3))
    ceiling_lights(sbox(-HW, -HC, HW, y0), CH)
    # stair hall walls run up with the stair; ceiling follows the slope 3 m above the nosings
    for s in (-1, 1):
        x = s * sw
        ob = B.mesh_obj(f"{CUR}_sw{s}", [(x, y0, 0.0), (x, y1 + 2.2, 0.0), (x, y1 + 2.2, zt + CH), (x, y1, zt + CH), (x, y0, CH)],
                        [(4, 3, 2, 1, 0) if s > 0 else (0, 1, 2, 3, 4)], M["dg_wall"],
                        {"UVMap": lambda co: (co[1] / UV_FLOOR, co[2] / UV_FLOOR)})
        put(ob)
    # stair: steps with yellow nosing strips, a top landing, a sloped soffit
    for i in range(steps):
        z = rise * (i + 1)
        yy = y0 + run * i
        bx(f"st{i}", (-sw, yy, 0.0), (sw, yy + run, z), M["stair"], deco=True)
        bx(f"sn{i}", (-sw + 0.05, yy, z - 0.002), (sw - 0.05, yy + 0.05, z + 0.003), M["nosing"], deco=True)
    bx("land", (-sw, y1, 0.0), (sw, y1 + 2.2, zt), M["stair"], deco=True)
    flat("landf", sbox(-sw, y1, sw, y1 + 2.2), zt + 0.001, M["dg_floor"], deco=True)
    ob = B.mesh_obj(f"{CUR}_soff", [(-sw, y0, CH), (sw, y0, CH), (sw, y1, zt + CH), (-sw, y1, zt + CH)],
                    [(3, 2, 1, 0)], M["dg_ceil"], {"UVMap": lambda co: (co[0] / UV_CEIL, co[1] / UV_CEIL)})
    put(ob)
    flat("ceil2", sbox(-sw, y1, sw, y1 + 2.2), zt + CH, M["dg_ceil"], up=False, uvs=UV_CEIL)
    # over the foot: the corridor ceiling meets the stair soffit (a bulkhead face)
    put(BU.vrect(f"{CUR}_bh", (-HW, y0), (HW, y0), CH, CH + 0.01, M["dg_ceil"], CUR, (0, -1, 0)))
    # handrails on both walls and down the middle
    for s, xx in ((-1, -sw + 0.06), (1, sw - 0.06), (0, 0.0)):
        p0 = Vector((xx, y0 - 0.3, 0.85))
        p1 = Vector((xx, y1 + 0.3, zt + 0.85))
        d = p1 - p0
        ob = B.cylinder(f"{CUR}_hr{s}", (0, 0, 0), 0.022, d.length, M["handrail_steel"], axis="Y", segs=12)
        ob.location = (p0 + p1) / 2
        ob.rotation_euler = (math.atan2(d.z, d.y), 0, 0)
        put(ob, deco=True)
        if s == 0:
            for t in np.linspace(0.05, 0.95, 6):
                q = p0 + d * t
                put(B.cylinder(f"{CUR}_hp{t:.2f}", (q.x, q.y, q.z - 0.43), 0.02, 0.86, M["handrail_steel"], segs=10), deco=True)
    # light rows along the slope and over the landing
    for i, t in enumerate(np.linspace(0.1, 0.9, 4)):
        yy = y0 + (y1 - y0) * t
        zz = CH + zt * t
        light_panel(f"sl{i}", (0.0, yy, zz - 0.002), LIGHT_W)
    # doors at the top: daylight behind glass, shut
    yd = y1 + 2.2
    put(BU.vrect(f"{CUR}_day", (-sw, yd + 0.5), (sw, yd + 0.5), zt, zt + CH, M["daylight"], CUR, (0, -1, 0)), deco=True)
    for s in (-1, 1):
        bx(f"df{s}", (s * sw - 0.1 if s > 0 else s * sw, yd, zt), (s * sw if s > 0 else s * sw + 0.1, yd + 0.1, zt + 2.3), M["steel_brushed"])
    bx("dtop", (-sw, yd, zt + 2.3), (sw, yd + 0.1, zt + CH), M["dg_wall"])
    for k, (xa, xb) in enumerate(((-sw + 0.1, -0.02), (0.02, sw - 0.1))):
        bx(f"dg{k}", (xa, yd + 0.03, zt), (xb, yd + 0.07, zt + 2.3), M["door_glass"])
        bx(f"dgf{k}", (xa, yd + 0.025, zt + 2.26), (xb, yd + 0.075, zt + 2.3), M["steel_brushed"], deco=True)
    ob = B.area_light(f"{CUR}_dayA", (0.0, yd + 0.4, zt + CH / 2), 2 * sw, CH, 900.0, (1.0, 0.98, 0.95),
                      rot=(math.radians(90), 0, 0), camera=False)
    put(ob, deco=True)
    hanging_sign("hs", (0.0, y0 - 1.2, 2.45), (0, -1), M["sign_exit"], w=2.4, h=0.48)
    # ramp the walker uses over the steps (not drawn)
    ramp(f"ramp", [(-sw, y0, 0.0), (sw, y0, 0.0), (sw, y1, zt), (-sw, y1, zt)])
    ramp(f"ramp2", [(-sw, y1, zt), (sw, y1, zt), (sw, yd, zt), (-sw, yd, zt)])


def ramp(name, pts):
    ob = B.mesh_obj(f"{CUR}_{name}", pts, [(0, 1, 2, 3)], None)
    ob["ramp"] = 1
    ob.hide_render = True
    put(ob)
    ob["nocol"] = 0
    return ob


# ------------------------------------------------------------------ materials

def materials():
    global M
    M = BU.materials()
    BU.M = M
    g = dgfx.build_all()
    gx = GX.build_all()
    for k, p in gx.items():
        M[k] = B.printed(k, p, rough=0.35, coat=0.6, emission=0.0)
    for k in ("poster_travel", "poster_museum", "poster_manners", "poster_ramen", "poster_mansion"):
        M[k] = B.printed(k, gx[k], rough=0.3, coat=0.0, emission=1.4)
    M["sign_exit"] = B.printed("sign_exit_lit", gx["sign_exit"], rough=0.3, coat=0.0, emission=2.0)
    for k, p in g.items():
        lit = k.startswith(("fascia_", "board_", "station_", "gate_", "mall_", "menu_cafe"))
        M[k] = B.printed(k, p, rough=0.35, coat=0.0, emission=2.2 if lit else 0.0)
    M["dg_floor"] = M["b1_floor"]
    M["dg_wall"] = M["wall_white"]
    M["dg_ceil"] = M["ceil"]
    M["glassbox_w"] = B.emissive("glassbox_w", (1.0, 0.88, 0.72), 6.0)
    M["door_grey"] = B.plain("door_grey", (0.42, 0.43, 0.44), 0.35, coat=0.2)
    M["door_glass"] = B.plain("door_glass", (0.70, 0.78, 0.78), 0.03, coat=1.0)
    M["door_glass"].node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.18
    M["daylight"] = B.emissive("daylight", (1.0, 0.99, 0.97), 12.0)
    M["vend_body"] = B.plain("vend_body", (0.80, 0.80, 0.80), 0.3, coat=0.5)
    M["vend_band"] = B.plain("vend_band", (0.75, 0.08, 0.08), 0.3, coat=0.5)
    M["vend_front"] = B.printed("vend_front", g["shelf_drinks"], rough=0.2, coat=0.8, emission=1.2)
    M["bench_wood"] = B.plain("bench_wood", (0.45, 0.28, 0.14), 0.45, coat=0.3)
    M["locker"] = B.plain("locker", (0.70, 0.72, 0.74), 0.3, metal=0.3)
    M["locker_door"] = B.plain("locker_door", (0.20, 0.38, 0.62), 0.3, coat=0.3)
    M["wood"] = B.plain("wood", (0.40, 0.25, 0.13), 0.45, coat=0.2)
    M["wood_light"] = B.plain("wood_light", (0.70, 0.55, 0.36), 0.45, coat=0.2)
    M["shop_glass"] = B.plain("shop_glass", (0.80, 0.86, 0.86), 0.02, coat=1.0)
    M["shop_glass"].node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.06
    M["shop_glass"].node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = 0.3
    M["shutter"] = B.plain("shutter", (0.62, 0.63, 0.64), 0.35, metal=0.6)
    M["shelf_white"] = B.plain("shelf_white", (0.85, 0.85, 0.84), 0.35)
    M["fridge"] = B.emissive("fridge", (0.95, 0.97, 1.0), 2.5)
    M["fridge_front"] = B.printed("fridge_front", g["shelf_drinks"], rough=0.1, coat=1.0, emission=1.5)
    M["counter_top"] = B.plain("counter_top", (0.20, 0.20, 0.21), 0.3, coat=0.4)
    M["tile_shop"] = M["b2_floor"]
    for k in ("shelf_mart", "shelf_mart2", "shelf_drinks", "shelf_drug", "shelf_books", "shelf_books2", "shelf_bread"):
        M[k] = B.printed(k + "_s", g[k], rough=0.4, coat=0.0, emission=0.0)
    import kit_goods
    kit_goods.materials(M)
    M["fabric"] = [B.plain(f"fabric{i}", c, 0.8) for i, c in enumerate(
        [(0.08, 0.10, 0.20), (0.55, 0.52, 0.47), (0.60, 0.15, 0.12), (0.85, 0.83, 0.78), (0.15, 0.30, 0.20), (0.05, 0.05, 0.05)])]
    M["flower"] = [B.plain(f"flower{i}", c, 0.7) for i, c in enumerate(
        [(0.80, 0.10, 0.25), (0.95, 0.75, 0.10), (0.90, 0.45, 0.60), (0.95, 0.95, 0.92), (0.50, 0.15, 0.60), (0.15, 0.45, 0.12)])]
    return M


def build_all(only=None):
    """Every module in its own collection (M_<name>) around the origin."""
    if not M:
        materials()
    for name, d in MODULES.items():
        if only and name not in only:
            continue
        begin(name)
        d["build"]()
        n = sum(1 for _ in BU.coll(CUR).objects)
        print(f"module {name}: {n} objects", flush=True)
    return MODULES


import kit_shops    # noqa: E402,F401  (registers the shop-street, plaza and station modules)
