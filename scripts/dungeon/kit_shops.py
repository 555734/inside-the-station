"""Shop streets, the plaza and the station of the maze (registered with kit.MODULES).

A shop street is a corridor cell whose side walls are shop fronts: two shops of 8 m frontage and
5 m depth on each side, each a whole little shop one can walk into (counter, shelves full of goods,
kitchen, tables...), lit from its own ceiling. The station is one module: the gate hall on the mall
level (2 x 2 cells) and, under it, the island platform the stairs lead down to.
"""
import math

import numpy as np
from mathutils import Vector
from shapely.geometry import box as sbox
from shapely.ops import unary_union

import build_scene as B
import build_umeda as BU
import kit as K
from kit import (C, HC, HW, CH, UV_FLOOR, UV_CEIL, LIGHT_W, Z2, E_, PORT, module, put, bx, quad,
                 Frame, wall_frame, flat, outline, walls, ceiling_lights, light_panel, tactile_line,
                 tactile_dots, wall_fixtures, ceiling_dressing, hanging_sign, ramp, _runs_by)

D = 5.0         # shop depth
W = 8.0         # shop frontage


def M():
    return K.M


# ------------------------------------------------------------------ one shop (front at y=0, inside y<0)

def shop(sid, tag):
    """A shop unit in its own frame: front along x in [0, W] at y=0, interior y in [-D, 0]."""
    kind = {s[0]: s[1] for s in K.dgfx.SHOPS}[sid]
    m = M()
    p = f"{tag}{sid}"
    pil = 0.3
    # pilasters (shared look with the corridor walls) and the fascia band with the lit name
    for k, (x0, x1) in enumerate(((0.0, pil), (W - pil, W))):
        bx(f"{p}pil{k}", (x0, -0.3, 0.0), (x1, 0.0, CH), m["dg_wall"])
    bx(f"{p}fas", (pil, -0.3, 2.45), (W - pil, 0.0, CH), m["clad_dark"])
    if kind == "shutter":
        # closed: a rolling shutter down to the floor, a notice taped on it
        bx(f"{p}sh", (pil, -0.25, 0.0), (W - pil, -0.2, 2.45), m["shutter"])
        for z in np.arange(0.12, 2.4, 0.10):
            bx(f"{p}shr{z:.2f}", (pil, -0.2, z), (W - pil, -0.185, z + 0.012), m["shutter"], deco=True)
        bx(f"{p}shb", (pil, -0.26, 2.30), (W - pil, -0.15, 2.45), m["steel_brushed"], deco=True)
        quad(f"{p}note", (W * 0.62, -0.183, 1.45), (-1, 0, 0), (0, 0, 1), 0.36, 0.51, m["shutter_notice"])
        return
    quad(f"{p}sign", (W / 2, 0.003, 2.72), (-1, 0, 0), (0, 0, 1), W - 2 * pil - 0.1, 0.48, m[f"fascia_{sid}"])
    # inside: floor, ceiling, side and back walls
    floor_mat = {"cafe": m["wood_light"], "bakery": m["wood_light"], "ramen": m["tile_shop"], "soba": m["tile_shop"],
                 "books": m["wood"], "clothes": m["wood_light"]}.get(kind, m["dg_floor"])
    flat(f"{p}fl", sbox(0.0, -D, W, -0.3), 0.0, floor_mat)
    flat(f"{p}fl2", sbox(pil, -0.3, W - pil, 0.0), 0.0, floor_mat)
    zc = 2.9
    flat(f"{p}ce", sbox(0.0, -D, W, -0.3), zc, m["panel_white"], up=False)
    put(BU.vrect(f"{K.CUR}_{p}ceb", (pil, -0.3), (W - pil, -0.3), 2.45, zc, m["panel_white"], K.CUR, (0, -1, 0)))
    wall_mat = {"cafe": m["wood"], "ramen": m["wood"], "soba": m["wood"]}.get(kind, m["panel_white"])
    # side walls 5 cm in from the shared line: the neighbour's wall stands back to back with this one
    # and would shade it black in the bake if both were on the same plane (the pilasters hide the gap)
    for a, b, n in (((0.05, -0.3), (0.05, -D), (1, 0)), ((W - 0.05, -D), (W - 0.05, -0.3), (-1, 0)), ((0.05, -D), (W - 0.05, -D), (0, 1))):
        put(BU.vrect(f"{K.CUR}_{p}w{a[0]:.0f}{b[1]:.0f}", a, b, 0.0, zc, wall_mat, K.CUR, (n[0], n[1], 0)))
    # front: glazing with mullions and an open entrance 2 m wide in the middle (doors slid open)
    ex0, ex1 = W / 2 - 1.0, W / 2 + 1.0
    open_front = kind in ("flowers", "bakery", "soba")
    if not open_front:
        for (x0, x1) in ((pil, ex0), (ex1, W - pil)):
            bx(f"{p}gl{x0:.0f}", (x0, -0.17, 0.10), (x1, -0.15, 2.40), m["shop_glass"])
            bx(f"{p}kick{x0:.0f}", (x0, -0.2, 0.0), (x1, -0.12, 0.10), m["steel_brushed"])
            for xm in np.arange(x0, x1 + 0.01, (x1 - x0) / max(1, round((x1 - x0) / 1.4))):
                bx(f"{p}mul{xm:.1f}", (xm - 0.025, -0.2, 0.0), (xm + 0.025, -0.12, 2.45), m["steel_brushed"], deco=True)
        bx(f"{p}tr", (pil, -0.2, 2.40), (W - pil, -0.12, 2.45), m["steel_brushed"], deco=True)
        # door leaves parked behind the glass beside the opening
        for s, xx in ((-1, ex0 - 0.5), (1, ex1 + 0.5)):
            bx(f"{p}leaf{s}", (xx - 0.5, -0.26, 0.02), (xx + 0.5, -0.24, 2.36), m["shop_glass"], deco=True)
    else:
        bx(f"{p}tr", (pil, -0.2, 2.40), (W - pil, -0.12, 2.45), m["steel_brushed"], deco=True)
    # lights
    warm = kind in ("cafe", "bakery", "ramen", "soba", "flowers")
    power = 70.0 if kind in ("konbini", "drug") else 50.0
    k = 0
    for x in np.arange(1.0, W, 2.0):
        for y in (-1.3, -3.3):
            put(B.cylinder(f"{K.CUR}_{p}dl{k}", (x, y, zc - 0.004), 0.07, 0.008, m["downlight"], segs=16), deco=True)
            ob = B.area_light(f"{K.CUR}_{p}A{k}", (x, y, zc - 0.02), 0.14, 0.14, power, (1.0, 0.86, 0.7) if warm else (1.0, 0.98, 0.95), camera=False)
            ob.data.spread = math.radians(140)
            put(ob, deco=True)
            k += 1
    FIT[kind](p, sid)


def shelf_unit(p, x0, x1, y0, y1, faces, h=1.5, rows=4, tex=("shelf_mart", "shelf_mart2"), top=True):
    """Gondola: a white shelf block with goods shown on its long faces ('+y', '-y')."""
    m = M()
    bx(f"{p}gb{x0:.1f}{y0:.1f}", (x0, y0, 0.0), (x1, y1, h), m["shelf_white"])
    for f in faces:
        yf = y1 + 0.003 if f == "+y" else y0 - 0.003
        right = (-1, 0, 0) if f == "+y" else (1, 0, 0)
        for r in range(rows):
            z = 0.12 + (h - 0.2) * (r + 0.5) / rows
            hh = (h - 0.2) / rows * 0.92
            quad(f"{p}gf{x0:.1f}{y0:.1f}{f}{r}", ((x0 + x1) / 2, yf, z), right, (0, 0, 1), x1 - x0 - 0.04, hh, m[tex[r % len(tex)]])


def wall_shelf(p, x0, x1, y, toward, h=2.0, rows=5, tex=("shelf_mart",), depth=0.45, name="ws"):
    """Shelves against a wall (at y, goods facing toward = +1/-1 along y)."""
    m = M()
    ya, yb = sorted((y, y + toward * depth))
    bx(f"{p}{name}{x0:.1f}", (x0, ya, 0.0), (x1, yb, h), m["shelf_white"])
    yf = y + toward * (depth + 0.003)
    right = (-1, 0, 0) if toward > 0 else (1, 0, 0)
    for r in range(rows):
        z = 0.12 + (h - 0.2) * (r + 0.5) / rows
        quad(f"{p}{name}f{x0:.1f}{r}", ((x0 + x1) / 2, yf, z), right, (0, 0, 1), x1 - x0 - 0.04, (h - 0.2) / rows * 0.92, m[tex[r % len(tex)]])


def counter(p, x0, x1, y0, y1, h=1.0, body="wood", top="counter_top"):
    m = M()
    bx(f"{p}ct{x0:.1f}{y0:.1f}", (x0, y0, 0.0), (x1, y1, h - 0.04), m[body], bevel=0.005)
    bx(f"{p}ctt{x0:.1f}{y0:.1f}", (x0 - 0.02, y0 - 0.02, h - 0.04), (x1 + 0.02, y1 + 0.02, h), m[top], deco=True)


def register(p, x, y):
    m = M()
    bx(f"{p}reg{x:.1f}", (x - 0.2, y - 0.15, 1.0), (x + 0.2, y + 0.15, 1.12), m["panel_grey"], deco=True)
    bx(f"{p}regs{x:.1f}", (x - 0.15, y - 0.02, 1.12), (x + 0.15, y + 0.0, 1.32), m["screen_bezel"], deco=True)


def stool(p, x, y, h=0.65):
    m = M()
    put(B.cylinder(f"{K.CUR}_{p}stl{x:.1f}{y:.1f}", (x, y, h / 2), 0.025, h, m["steel"], segs=10), deco=True)
    put(B.cylinder(f"{K.CUR}_{p}sts{x:.1f}{y:.1f}", (x, y, h), 0.17, 0.05, m["counter_top"], segs=20))


def table(p, x, y, r=0.35, h=0.72, chairs=2):
    m = M()
    put(B.cylinder(f"{K.CUR}_{p}tt{x:.1f}{y:.1f}", (x, y, h - 0.02), r, 0.04, m["wood"], segs=24))
    put(B.cylinder(f"{K.CUR}_{p}tl{x:.1f}{y:.1f}", (x, y, (h - 0.04) / 2), 0.04, h - 0.04, m["steel"], segs=10), deco=True)
    for i in range(chairs):
        a = math.pi * 2 * i / chairs + 0.3
        cx, cy = x + math.cos(a) * 0.6, y + math.sin(a) * 0.6
        bx(f"{p}ch{x:.1f}{y:.1f}{i}", (cx - 0.2, cy - 0.2, 0.42), (cx + 0.2, cy + 0.2, 0.46), m["wood"])
        for lx in (-0.16, 0.16):
            for ly in (-0.16, 0.16):
                bx(f"{p}chl{x:.1f}{y:.1f}{i}{lx}{ly}", (cx + lx - 0.015, cy + ly - 0.015, 0.0), (cx + lx + 0.015, cy + ly + 0.015, 0.42), m["steel"], deco=True)


def fit_konbini(p, sid):
    m = M()
    # drinks fridges along the back wall (lit inside), gondolas, a counter with registers by the door
    bx(f"{p}frb", (0.3, -D, 0.0), (W - 0.3, -D + 0.75, 2.1), m["shelf_white"])
    for i, x in enumerate(np.arange(0.3, W - 0.31, 0.9)):
        quad(f"{p}frg{i}", (x + 0.45, -D + 0.753, 1.05), (-1, 0, 0), (0, 0, 1), 0.84, 1.75, m["fridge_front"])
        bx(f"{p}frh{i}", (x + 0.7, -D + 0.755, 0.9), (x + 0.73, -D + 0.80, 1.3), m["steel"], deco=True)
    for j, yy in enumerate((-3.3, -2.0)):
        shelf_unit(f"{p}g{j}", 2.6, 6.8, yy - 0.3, yy + 0.3, ("+y", "-y"))
    counter(p, 0.4, 2.4, -1.5, -0.9, body="shelf_white")
    register(p, 0.9, -1.2)
    register(p, 1.9, -1.2)
    wall_shelf(p, 0.05, 0.5, -3.0, 1, h=1.6, rows=4, tex=("shelf_mart2",), depth=0.4, name="ms")


def fit_cafe(p, sid):
    m = M()
    counter(p, 1.0, 7.0, -D + 0.3, -D + 1.0, body="wood")
    bx(f"{p}esp", (5.5, -D + 0.45, 1.0), (6.3, -D + 0.85, 1.45), m["steel"], deco=True)
    quad(f"{p}menu", (4.0, -D + 0.01, 2.2), (-1, 0, 0), (0, 0, 1), 2.4, 0.6, m["menu_cafe"])
    for i, (x, y) in enumerate(((1.3, -2.4), (3.3, -2.6), (6.4, -2.4), (1.3, -1.2), (6.6, -1.2))):
        table(p, x, y)
    for x in (2.0, 4.0, 6.0):
        put(B.cylinder(f"{K.CUR}_{p}pd{x}", (x, -D + 0.7, 2.2), 0.12, 0.25, m["glassbox_w"], segs=16, r2=0.05), deco=True)
        put(B.cylinder(f"{K.CUR}_{p}pdc{x}", (x, -D + 0.7, 2.6), 0.006, 0.62, m["handrail"], segs=6), deco=True)


def fit_bakery(p, sid):
    m = M()
    for k, x0 in enumerate((0.4, 4.3)):
        wall_shelf(p, x0, x0 + 3.3, -D, 1, h=1.8, rows=4, tex=("shelf_bread",), depth=0.5, name=f"bs{k}")
    bx(f"{p}tbl", (2.6, -3.0, 0.0), (5.4, -2.0, 0.85), m["wood"])
    for k, x in enumerate((2.9, 3.6, 4.3, 5.0)):
        quad(f"{p}tray{k}", (x + 0.15, -2.5, 0.852), (1, 0, 0), (0, 1, 0), 0.6, 0.8, m["shelf_bread"])
    counter(p, 5.6, 7.6, -1.6, -0.9, body="wood")
    register(p, 6.6, -1.25)


def fit_drug(p, sid):
    for j, yy in enumerate((-3.6, -2.2)):
        shelf_unit(f"{p}g{j}", 1.2, 6.8, yy - 0.3, yy + 0.3, ("+y", "-y"), h=1.6, tex=("shelf_drug", "shelf_mart"))
    wall_shelf(p, 0.4, W - 0.4, -D, 1, h=2.0, rows=5, tex=("shelf_drug", "shelf_mart2"), depth=0.4)
    counter(p, 0.4, 1.0, -3.0, -1.0, body="shelf_white")


def fit_books(p, sid):
    m = M()
    wall_shelf(p, 0.3, W - 0.3, -D, 1, h=2.1, rows=6, tex=("shelf_books", "shelf_books2"), depth=0.35, name="bk")
    for j, yy in enumerate((-3.2,)):
        shelf_unit(f"{p}g{j}", 1.5, 6.5, yy - 0.3, yy + 0.3, ("+y", "-y"), h=1.4, rows=4, tex=("shelf_books", "shelf_books2"))
    rng = np.random.default_rng(len(p))
    for t, (x0, x1) in enumerate(((1.4, 3.2), (4.8, 6.6))):
        bx(f"{p}tb{t}", (x0, -2.0, 0.0), (x1, -1.3, 0.8), m["wood"])
        for i in range(8):
            cx = x0 + 0.2 + (x1 - x0 - 0.4) * (i % 4) / 3
            cy = -1.85 + 0.4 * (i // 4)
            hh = 0.03 * int(rng.integers(2, 6))
            bx(f"{p}bs{t}{i}", (cx - 0.1, cy - 0.14, 0.8), (cx + 0.1, cy + 0.14, 0.8 + hh), m["fabric"][int(rng.integers(len(m["fabric"])))], deco=True)
    counter(p, 6.8, 7.7, -3.6, -2.2, body="wood")


def fit_ramen(p, sid):
    m = M()
    # kitchen behind an L-shaped counter, stools in front of it, menu tags over the kitchen
    bx(f"{p}kit", (0.3, -D, 0.0), (W - 0.3, -D + 0.7, 0.9), m["steel_brushed"])
    bx(f"{p}hood", (0.5, -D, 2.1), (W - 0.5, -D + 0.6, 2.6), m["steel_brushed"], deco=True)
    counter(p, 0.3, W - 0.3, -3.2, -2.6, h=1.0, body="wood")
    counter(p, 0.3, 0.9, -2.6, -1.0, h=1.0, body="wood")
    for x in np.arange(1.6, W - 0.5, 0.75):
        stool(p, x, -2.2)
    quad(f"{p}menu", (4.0, -D + 0.62, 2.35), (-1, 0, 0), (0, 0, 1), 2.4, 0.6, m["menu_ramen"])
    quad(f"{p}noren", (4.0, 0.02, 2.05), (-1, 0, 0), (0, 0, 1), 2.0, 0.8, m["noren_ramen"])
    bx(f"{p}tkt", (W - 1.3, -0.9, 0.0), (W - 0.6, -0.4, 1.7), m["vend_body"])


def fit_soba(p, sid):
    m = M()
    bx(f"{p}kit", (0.3, -D, 0.0), (W - 0.3, -D + 0.8, 0.9), m["steel_brushed"])
    counter(p, 0.3, W - 0.3, -3.6, -3.1, h=1.05, body="wood")
    counter(p, 0.3, 0.8, -3.1, -0.6, h=1.05, body="wood")
    counter(p, W - 0.8, W - 0.3, -3.1, -0.6, h=1.05, body="wood")
    quad(f"{p}menu", (4.0, -D + 0.02, 2.3), (-1, 0, 0), (0, 0, 1), 2.4, 0.6, m["menu_soba"])
    quad(f"{p}noren", (4.0, 0.02, 2.05), (-1, 0, 0), (0, 0, 1), 2.6, 0.8, m["noren_soba"])


def fit_clothes(p, sid):
    m = M()
    rng = np.random.default_rng(7)
    for r, (x0, x1, y) in enumerate(((0.6, 3.6, -3.8), (4.4, 7.4, -3.8), (2.0, 6.0, -2.2))):
        bx(f"{p}rk{r}", (x0, y - 0.02, 1.45), (x1, y + 0.02, 1.49), m["steel"], deco=True)
        for s in (x0, x1):
            bx(f"{p}rkp{r}{s:.1f}", (s - 0.02, y - 0.02, 0.0), (s + 0.02, y + 0.02, 1.49), m["steel"])
        for i, x in enumerate(np.arange(x0 + 0.1, x1 - 0.05, 0.12)):
            c = m["fabric"][int(rng.integers(len(m["fabric"])))]
            bx(f"{p}cl{r}{i}", (x, y - 0.25, 0.75), (x + 0.04, y + 0.25, 1.47), c, deco=True)
    wall_shelf(p, 0.4, W - 0.4, -D, 1, h=2.0, rows=4, tex=("shelf_mart",), depth=0.3, name="cs")
    put(B.cylinder(f"{K.CUR}_{p}mqb", (7.2, -1.2, 0.015), 0.22, 0.03, m["steel"], segs=20))
    put(B.cylinder(f"{K.CUR}_{p}mqp", (7.2, -1.2, 0.47), 0.02, 0.9, m["steel"], segs=8), deco=True)
    put(B.cylinder(f"{K.CUR}_{p}mq", (7.2, -1.2, 1.17), 0.18, 0.55, m["kiosk_white"], segs=16))
    put(B.sphere(f"{K.CUR}_{p}mqh", (7.2, -1.2, 1.55), 0.11, m["kiosk_white"]), deco=True)
    bx(f"{p}mir", (0.01, -2.5, 0.3), (0.03, -1.5, 2.0), m["mirror"], deco=True)


def fit_flowers(p, sid):
    m = M()
    rng = np.random.default_rng(11)
    for k, (x0, y0) in enumerate(((0.6, -1.2), (2.2, -1.0), (5.4, -1.0), (6.8, -1.2), (1.0, -4.3), (3.0, -4.3), (5.0, -4.3), (7.0, -4.3))):
        put(B.cylinder(f"{K.CUR}_{p}bk{k}", (x0, y0, 0.25), 0.22, 0.5, m["steel_brushed"], segs=20))
        put(B.cylinder(f"{K.CUR}_{p}lv{k}", (x0, y0, 0.66), 0.2, 0.34, m["flower"][5], segs=16, r2=0.14), deco=True)
        for j in range(7):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, 0.16)
            put(B.sphere(f"{K.CUR}_{p}fl{k}{j}", (x0 + math.cos(a) * r * 0.7, y0 + math.sin(a) * r * 0.7, 0.84 + rng.uniform(0, 0.04)), 0.07,
                         m["flower"][int(rng.integers(len(m["flower"])))], segs=10), deco=True)
    wall_shelf(p, 0.4, W - 0.4, -D, 1, h=1.6, rows=3, tex=("shelf_mart2",), depth=0.4, name="fs")
    counter(p, 3.0, 5.0, -3.2, -2.6, body="wood")


FIT = {"konbini": fit_konbini, "cafe": fit_cafe, "bakery": fit_bakery, "drug": fit_drug, "books": fit_books,
       "ramen": fit_ramen, "soba": fit_soba, "clothes": fit_clothes, "flowers": fit_flowers}


def shop_street(east, west):
    """A street cell (N-S) with two shops on each side; east/west = (south shop, north shop)."""
    m = M()
    poly = sbox(-HW, -HC, HW, HC)
    flat("floor", poly, 0.0, m["dg_floor"])
    flat("ceil", poly, CH, m["dg_ceil"], up=False, uvs=UV_CEIL)
    tactile_line("tl", (0, -HC), (0, HC))
    ceiling_lights(poly, CH)
    for k, sid in enumerate(east):
        y0 = -HC + k * W
        with wall_frame((HW, y0), (-1, 0)):
            shop(sid, f"e{k}")
        # projecting sign board out from the pilaster, seen along the street
        put(B.box(f"{K.CUR}_pb_e{k}", (HW - 0.38, y0 + W - 0.15, 2.65), (0.6, 0.06, 0.6), m["clad_dark"]), deco=True)
        bx(f"pbk_e{k}", (HW - 0.09, y0 + W - 0.17, 2.80), (HW, y0 + W - 0.13, 2.86), m["steel"], deco=True)
        for s in (-1, 1):
            put(B.image_quad(f"{K.CUR}_pbf_e{k}{s}", (HW - 0.38, y0 + W - 0.15 + 0.031 * s, 2.65), (-s, 0, 0), (0, 0, 1), 0.56, 0.56, m[f"board_{sid}"] if f"board_{sid}" in m else m["clad_dark"]), deco=True)
    for k, sid in enumerate(west):
        y1 = -HC + (k + 1) * W
        with wall_frame((-HW, y1), (1, 0)):
            shop(sid, f"w{k}")
        put(B.box(f"{K.CUR}_pb_w{k}", (-HW + 0.38, y1 - W + 0.15, 2.65), (0.6, 0.06, 0.6), m["clad_dark"]), deco=True)
        bx(f"pbk_w{k}", (-HW, y1 - W + 0.13, 2.80), (-HW + 0.09, y1 - W + 0.17, 2.86), m["steel"], deco=True)
        for s in (-1, 1):
            put(B.image_quad(f"{K.CUR}_pbf_w{k}{s}", (-HW + 0.38, y1 - W + 0.15 + 0.031 * s, 2.65), (s, 0, 0), (0, 0, 1), 0.56, 0.56, m[f"board_{sid}"] if f"board_{sid}" in m else m["clad_dark"]), deco=True)
    ceiling_dressing([("smoke", 0.0, -4.0), ("cam", -1.5, 4.0)])


@module("SHOP_A", [(0, 0)], [(0, HC, "N"), (0, -HC, "S")])
def build_shop_a():
    shop_street(east=("mart", "cafe"), west=("bake", "drug"))


@module("SHOP_B", [(0, 0)], [(0, HC, "N"), (0, -HC, "S")])
def build_shop_b():
    shop_street(east=("ramen", "book"), west=("wear", "shut"))


@module("SHOP_C", [(0, 0)], [(0, HC, "N"), (0, -HC, "S")])
def build_shop_c():
    shop_street(east=("soba", "flower"), west=("shut", "mart"))


# ------------------------------------------------------------------ plaza (2 x 2 cells)

PZ = 13.0       # half size of the plaza hall
PZ_H = 4.2


def stub_rects(ports):
    """Corridor stubs from the hall edge (+-PZ) to the 2x2 module edge (+-C) at the given ports."""
    out = []
    for (x, y, side) in ports:
        if side in ("N", "S"):
            y0, y1 = sorted((math.copysign(PZ, y), y))
            out.append((x - HW, y0, x + HW, y1))
        else:
            x0, x1 = sorted((math.copysign(PZ, x), x))
            out.append((x0, y - HW, x1, y + HW))
    return out


def port_box(x, y, side):
    if side in ("N", "S"):
        return (x - HW + E_, y - 0.05, x + HW - E_, y + 0.05)
    return (x - 0.05, y - HW + E_, x + 0.05, y + HW - E_)


def hall(ports, h=PZ_H, floor="dg_floor", hole=None):
    """Hall of +-PZ with corridor stubs to the ports; the hall ceiling is higher than the stubs'
    (a bulkhead closes the step). hole: a box cut out of the floor (stairwell)."""
    m = M()
    hall_p = sbox(-PZ, -PZ, PZ, PZ)
    stubs = [sbox(*r) for r in stub_rects(ports)]
    fl = unary_union([hall_p] + stubs)
    if hole:
        fl = fl.difference(sbox(*hole))
    flat("floor", fl, 0.0, m[floor])
    flat("ceil", hall_p, h, m["dg_ceil"], up=False, uvs=UV_CEIL)
    for k, s in enumerate(stubs):
        flat(f"sceil{k}", s, CH, m["dg_ceil"], up=False, uvs=UV_CEIL)
    whole = unary_union([hall_p] + stubs)
    runs = outline(whole, [port_box(*p) for p in ports])
    # walls: stubs CH high, hall walls h high
    hr = [r for r in runs if max(abs(r[0][0]), abs(r[0][1]), abs(r[1][0]), abs(r[1][1])) <= PZ + 0.01
          and (abs(abs(r[0][0]) - PZ) < 0.01 and abs(abs(r[1][0]) - PZ) < 0.01 or abs(abs(r[0][1]) - PZ) < 0.01 and abs(abs(r[1][1]) - PZ) < 0.01)]
    sr = [r for r in runs if r not in hr]
    walls("hw", hr, 0.0, h, m["dg_wall"])
    walls("sw", sr, 0.0, CH, m["dg_wall"])
    # bulkheads over the stub mouths (from the stub ceiling up to the hall ceiling)
    for k, (x, y, side) in enumerate(ports):
        if side in ("N", "S"):
            yy = math.copysign(PZ, y)
            put(BU.vrect(f"{K.CUR}_bk{k}", (x - HW, yy), (x + HW, yy), CH, h, m["dg_wall"], K.CUR, (0, -math.copysign(1, y), 0)))
        else:
            xx = math.copysign(PZ, x)
            put(BU.vrect(f"{K.CUR}_bk{k}", (xx, y - HW), (xx, y + HW), CH, h, m["dg_wall"], K.CUR, (-math.copysign(1, x), 0, 0)))
    ceiling_lights(hall_p, h, margin=1.2, power=LIGHT_W * 1.2)
    for k, s in enumerate(stubs):
        ceiling_lights(s, CH, name=f"Ls{k}")
    return runs, hr, sr


PLAZA_PORTS = [(-8.0, C, "N"), (C, 8.0, "E"), (8.0, -C, "S"), (-C, -8.0, "W")]


CELLS2 = [(-HC, -HC), (HC, -HC), (-HC, HC), (HC, HC)]


@module("PLAZA", CELLS2, PLAZA_PORTS)
def build_plaza():
    """The mall's plaza: a high hall with a round lit pillar ringed by a bench in the middle, four
    big square columns with posters, a mall banner, benches and planters."""
    m = M()
    runs, hr, sr = hall(PLAZA_PORTS)
    # central round pillar with a glowing band and a ring bench
    put(B.cylinder(f"{K.CUR}_cp", (0, 0, PZ_H / 2), 1.2, PZ_H, m["column"], segs=48))
    put(B.cylinder(f"{K.CUR}_cpb", (0, 0, 2.6), 1.23, 0.9, m["led_ring"], segs=48), deco=True)
    put(B.cylinder(f"{K.CUR}_rb", (0, 0, 0.22), 2.4, 0.44, m["bench_wood"], segs=48))
    put(B.cylinder(f"{K.CUR}_rbi", (0, 0, 0.23), 1.9, 0.46, m["column"], segs=48))
    # four columns, posters on their faces
    for k, (cx, cy) in enumerate(((-7, -7), (7, -7), (-7, 7), (7, 7))):
        bx(f"col{k}", (cx - 0.6, cy - 0.6, 0.0), (cx + 0.6, cy + 0.6, PZ_H), m["clad_dark"])
        for j, (n, r) in enumerate((((1, 0), (0, 1)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)), ((0, -1), (1, 0)))):
            c = (cx + n[0] * 0.603, cy + n[1] * 0.603, 1.6)
            quad(f"colp{k}{j}", c, (r[0], r[1], 0), (0, 0, 1), 1.0, 1.2, m[f"adp{(k + j) % 3}"])
    # mall banner over two hall walls, benches and planters along them
    for k, (x, y, n) in enumerate(((0.0, PZ, (0, -1)), (0.0, -PZ, (0, 1)))):
        r = (-n[1], n[0], 0) if False else ((1, 0, 0) if n[1] < 0 else (-1, 0, 0))
        quad(f"ban{k}", (x + (3.5 if n[1] < 0 else -3.5), y + n[1] * 0.02, 3.3), r, (0, 0, 1), 4.0, 1.0, m["mall_banner"])
    for k, (x, y) in enumerate(((-10.5, 4.0), (10.5, -4.0))):
        bx(f"pl{k}", (x - 0.6, y - 1.6, 0.0), (x + 0.6, y + 1.6, 0.6), m["clad_dark"])
        for j in range(6):
            put(B.sphere(f"{K.CUR}_plg{k}{j}", (x + ((j % 2) - 0.5) * 0.5, y - 1.2 + (j // 2) * 1.2, 0.85), 0.35, m["flower"][5], segs=12), deco=True)
    for k, (x, y, a) in enumerate(((-4.0, -10.6, 0), (4.0, 10.6, 180), (-10.6, -3.0, -90))):
        with Frame((x, y), a):
            K.bench(f"bn{k}", 0.0, 2.4)
    ceiling_dressing([("cam", 5.0, 0.0), ("cam", -5.0, 0.0), ("smoke", 0.0, 6.0), ("smoke", 0.0, -6.0)], z=PZ_H)


# ------------------------------------------------------------------ the station (gate hall + platform)

GATE_PORTS = [(-8.0, C, "N"), (8.0, C, "N"), (-C, 8.0, "W"), (C, 8.0, "E")]
GH = 3.6                 # gate hall ceiling
GATE_Y = -1.0            # gate line
SW = (-8.0, -13.0, 8.0, -7.0)     # stairwell opening in the hall floor (x0, y0, x1, y1)
PL = (-48.0, 48.0)       # platform x range
PY = (-15.0, -5.0)       # platform y range
TRACK_W = 4.5
PCH = 4.0                # platform ceiling height


@module("STATION", CELLS2, GATE_PORTS)
def build_station():
    gate_hall()
    platform()
    station_stair()


def gate_hall():
    m = M()
    runs, hr, sr = hall(GATE_PORTS, h=GH, hole=SW)
    # gate line: 7 cabinets (6 passages) left of the middle, the office window, fences to the walls
    n = 7
    xs = [-5.0 + i * 1.0 for i in range(n)]
    for i, x in enumerate(xs):
        BU.Z_B1 = 0.0
        BU._gate_cabinet(f"{K.CUR}_g{i}", Vector((x, GATE_Y, 0.0)), math.pi / 2, 1, i, n - 1, m)
    for o in list(BU.coll("B1").objects):          # _gate_cabinet puts its parts in B1
        BU.put(o, K.CUR)
    BU.gate_office(f"{K.CUR}_off", (2.6, GATE_Y), (6.0, GATE_Y), (0, 1), m, cname=K.CUR)
    BU.gate_fence(f"{K.CUR}_fw", (-PZ, GATE_Y), (xs[0] - 0.14, GATE_Y), m, cname=K.CUR)
    BU.gate_fence(f"{K.CUR}_fm", (xs[-1] + 0.14, GATE_Y), (2.6, GATE_Y), m, cname=K.CUR)
    BU.gate_fence(f"{K.CUR}_fe", (6.0, GATE_Y), (PZ, GATE_Y), m, cname=K.CUR)
    hanging_sign("gs", (-1.0, GATE_Y + 0.6, 2.9), (0, 1), m["gate_a"], w=4.8, h=0.4, ceil=GH)
    # stairwell balustrade (glass on a steel base) on the open sides of the opening
    x0, y0, x1, y1 = SW
    for k, (a, b) in enumerate((((x0, y1), (x1, y1)), ((x1, y0), (x1, y1)))):
        lo = (min(a[0], b[0]) - 0.03, min(a[1], b[1]) - 0.03, 0.0)
        hi = (max(a[0], b[0]) + 0.03, max(a[1], b[1]) + 0.03, 0.15)
        bx(f"bb{k}", lo, hi, m["steel_brushed"])
        bx(f"bg{k}", (lo[0] + 0.015, lo[1] + 0.015, 0.15), (hi[0] - 0.015, hi[1] - 0.015, 1.05), m["glass_rail"])
        bx(f"bt{k}", (lo[0], lo[1], 1.05), (hi[0], hi[1], 1.10), m["steel_brushed"], deco=True)
    # stairwell walls down to the platform ceiling level (inside the slab)
    for k, (a, b, nn) in enumerate((((x0, y0), (x1, y0), (0, 1)), ((x1, y1), (x0, y1), (0, -1)), ((x1, y0), (x1, y1), (-1, 0)), ((x0, y1), (x0, y0), (1, 0)))):
        put(BU.vrect(f"{K.CUR}_sww{k}", a, b, Z2 + PCH, 0.0, m["dg_wall"], K.CUR, (nn[0], nn[1], 0)))
    hanging_sign("ps", (x0 + 1.2, -4.0, 2.6), (-1, 0), m["mall_sign"], w=2.4, h=0.4, ceil=GH)
    wall_fixtures(runs, {_runs_by(runs, (0, -1), (0, PZ)): [("poster", 2.5, "adp0"), ("poster", 7.5, "adp2")]})


def platform():
    """Island platform (Z2) with half-height platform doors, tracks behind them, tiled track walls
    with station name boards, columns down the middle, lights in rows."""
    m = M()
    x0, x1 = PL
    ya, yb = PY
    zf = Z2
    zc = Z2 + PCH
    tw_s, tw_n = ya - TRACK_W, yb + TRACK_W
    plat = sbox(x0, ya, x1, yb).difference(sbox(SW[0], SW[1], SW[2] - 0.8, SW[3]))
    with Frame((0, 0, zf), 0):
        flat("pfl", plat, 0.0, m["b2_floor"])
    # ceiling over platform and tracks
    with Frame((0, 0, zc), 0):
        flat("pce", sbox(x0, tw_s, x1, tw_n).difference(sbox(*SW)), 0.0, m["ceil_b2"], up=False, uvs=UV_CEIL)
    # track beds and walls
    for k, (y0, y1, wall_y, nn) in enumerate(((tw_s, ya, tw_s, 1), (yb, tw_n, tw_n, -1))):
        with Frame((0, 0, zf - 1.1), 0):
            flat(f"tb{k}", sbox(x0 - 16, y0, x1 + 16, y1), 0.0, m["concrete"])
        # platform edge face down to the track bed
        ye = ya if k == 0 else yb
        put(BU.vrect(f"{K.CUR}_pe{k}", (x0, ye), (x1, ye), zf - 1.1, zf, m["concrete"], K.CUR, (0, -nn, 0)))
        put(BU.vrect(f"{K.CUR}_tw{k}", (x0 - 16, wall_y), (x1 + 16, wall_y), zf - 1.1, zc, m["track_tile"], K.CUR, (0, nn, 0), 2.16))
        for r, yr in enumerate((y0 + 0.9, y1 - 0.9)):
            bx(f"rl{k}{r}", (x0 - 16, yr - 0.035, zf - 1.1), (x1 + 16, yr + 0.035, zf - 0.95), m["rail"], deco=True)
        for xs_ in np.arange(x0 - 15.5, x1 + 16, 0.6):
            bx(f"sl{k}{xs_:.1f}", (xs_ - 0.1, y0 + 0.5, zf - 1.1), (xs_ + 0.1, y1 - 0.5, zf - 1.04), m["concrete"], deco=True)
        # station name boards and ads on the track wall
        for j, xx in enumerate(np.arange(x0 + 8, x1, 16.0)):
            c = (xx, wall_y + nn * 0.027, zf + 2.2)
            bx(f"nbf{k}{j}", (xx - 1.55, min(wall_y, wall_y + nn * 0.025), zf + 1.8), (xx + 1.55, max(wall_y, wall_y + nn * 0.025), zf + 2.6), m["clad_dark"], deco=True)
            quad(f"nb{k}{j}", c, ((-1, 0, 0) if nn > 0 else (1, 0, 0)), (0, 0, 1), 3.0, 0.7, m["station_a"])
            for s in (-5.0, 5.0):
                cc = (xx + s, wall_y + nn * 0.027, zf + 1.9)
                bx(f"adf{k}{j}{s:.0f}", (xx + s - 1.41, min(wall_y, wall_y + nn * 0.025), zf + 1.09), (xx + s + 1.41, max(wall_y, wall_y + nn * 0.025), zf + 2.71), m["clad_dark"], deco=True)
                quad(f"ad{k}{j}{s:.0f}", cc, ((-1, 0, 0) if nn > 0 else (1, 0, 0)), (0, 0, 1), 2.7, 1.5, m[f"ad{(j + k) % 3}"])
        # track tunnels beyond the platform ends (dark) and their end walls
        for s in (-1, 1):
            xe = x0 - 16 if s < 0 else x1 + 16
            put(BU.vrect(f"{K.CUR}_te{k}{s}", (xe, y0), (xe, y1), zf - 1.1, zc, m["concrete"], K.CUR, (-s, 0, 0)))
            with Frame((0, 0, zc), 0):
                flat(f"tce{k}{s}", sbox(min(xe, x0 if s < 0 else x1), y0, max(xe, x0 if s < 0 else x1), y1), 0.0, m["concrete"], up=False)
        # platform doors along the edge: fixed panels with door pairs every 4.5 m, shut
        ye = ya if k == 0 else yb
        for j, xx in enumerate(np.arange(x0 + 1.5, x1 - 1.0, 4.5)):
            bx(f"pd{k}{j}", (xx, ye - 0.08 if nn > 0 else ye, zf), (xx + 3.0, ye if nn > 0 else ye + 0.08, zf + 1.3), m["psd_panel"])
            bx(f"pdd{k}{j}", (xx + 3.0, ye - 0.06 if nn > 0 else ye + 0.02, zf), (xx + 4.5, ye - 0.02 if nn > 0 else ye + 0.06, zf + 1.3), m["psd_glass"])
        # tactile warning line inside the doors
        tl_y = ye + nn * 0.5
        with Frame((0, 0, zf), 0):
            tactile_line(f"ptl{k}", (x0 + 0.3, tl_y), (x1 - 0.3, tl_y))
    # end walls of the platform and the walls between platform/track ends
    for s in (-1, 1):
        xe = x0 if s < 0 else x1
        put(BU.vrect(f"{K.CUR}_pew{s}", (xe, ya), (xe, yb), zf, zc, m["wall_white"], K.CUR, (-s, 0, 0)))
        with Frame((xe, (ya + yb) / 2, zf), 90 if s > 0 else -90):
            K.staff_door(f"psd{s}", 0.0)
    # columns along the middle (not in the stair zone) with benches between
    for j, xx in enumerate(np.arange(x0 + 6, x1 - 5, 8.0)):
        if SW[0] - 1.5 < xx < SW[2] + 3.0:
            continue
        bx(f"pc{j}", (xx - 0.35, -10.35, zf), (xx + 0.35, -9.65, zc), m["column"])
        with Frame((xx + 2.0, -10.0, zf), 0):
            K.bench(f"pb{j}", 0.0, 1.8)
    # lights in two rows over the platform, dimmer rows over the tracks
    for j, xx in enumerate(np.arange(x0 + 2.0, x1, 4.0)):
        for r, yy in enumerate((-12.5, -7.5)):
            if SW[0] - 0.5 < xx < SW[2] and SW[1] < yy < SW[3]:
                continue
            light_panel(f"pl{j}{r}", (xx, yy, zc - 0.002), LIGHT_W * 1.1, size=(1.2, 0.3))
        for r, yy in enumerate((tw_s + 2.2, tw_n - 2.2)):
            light_panel(f"tl{j}{r}", (xx, yy, zc - 0.002), LIGHT_W * 0.35, size=(1.2, 0.15))
    # hanging direction signs and departure boards
    for k, xx in enumerate((-30.0, 30.0)):
        hanging_sign_at(f"dp{k}", (xx, -10.0, zf + 2.75), (1, 0) if xx < 0 else (-1, 0), m["mall_sign"], zc)


def hanging_sign_at(name, c, facing, mat, zc, w=2.4, h=0.4):
    """hanging_sign for a ceiling other than the corridor's."""
    old = K.CH
    K.CH = zc
    try:
        hanging_sign(name, c, facing, mat, w, h)
    finally:
        K.CH = old


def station_stair():
    """Stair from the paid side of the gate hall (west end of the opening, z=0) down to the platform
    (east), with a landing half way; the walls of the opening run down to the platform."""
    m = M()
    x0, y0, x1, y1 = SW
    rise = 7.0 / 44
    run = 0.30
    xa = x0 + 0.2
    pts = []
    z = 0.0
    x = xa
    for i in range(44):
        if i == 22:
            bx("lnd", (x, y0, Z2), (x + 2.0, y1, z), m["stair"], deco=True)
            pts.append((x, z))
            x += 2.0
            pts.append((x, z))
        z -= rise
        bx(f"st{i}", (x, y0, Z2), (x + run, y1, z), m["stair"], deco=True)
        bx(f"sn{i}", (x + run - 0.05, y0 + 0.05, z - 0.002), (x + run, y1 - 0.05, z + 0.003), m["nosing"], deco=True)
        if i == 0:
            pts.insert(0, (x, 0.0))
        x += run
    pts.append((x, Z2))
    xb = x
    # walls on both sides of the stair from the platform up to the hall floor, end wall under the top
    for k, (yy, nn) in enumerate(((y0, 1), (y1, -1))):
        put(BU.vrect(f"{K.CUR}_sws{k}", (x0, yy), (xb + 0.01, yy), Z2, Z2 + PCH, m["dg_wall"], K.CUR, (0, nn, 0)))
        put(BU.vrect(f"{K.CUR}_swo{k}", (x0, yy + (-0.01 if nn > 0 else 0.01)), (xb, yy + (-0.01 if nn > 0 else 0.01)), Z2, Z2 + PCH, m["dg_wall"], K.CUR, (0, -nn, 0)))
    put(BU.vrect(f"{K.CUR}_swe", (x0 - 0.01, y0), (x0 - 0.01, y1), Z2, Z2 + PCH, m["dg_wall"], K.CUR, (-1, 0, 0)))
    # handrails along both walls
    for k, yy in enumerate((y0 + 0.06, y1 - 0.06)):
        for a, b in zip(pts[:-1], pts[1:]):
            p0 = Vector((a[0], yy, a[1] + 0.85))
            p1 = Vector((b[0], yy, b[1] + 0.85))
            d = p1 - p0
            ob = B.cylinder(f"{K.CUR}_hr{k}{a[0]:.1f}", (0, 0, 0), 0.022, d.length, m["handrail_steel"], axis="X", segs=12)
            ob.location = (p0 + p1) / 2
            ob.rotation_euler = (0, -math.atan2(d.z, d.x), 0)
            put(ob, deco=True)
    # ramp the walker uses over the steps
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        ramp(f"ramp{k}", [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])])
    hanging_sign_at("pfs", (xb + 1.0, -10.0, Z2 + 2.75), (1, 0), m["mall_sign"], Z2 + PCH)
