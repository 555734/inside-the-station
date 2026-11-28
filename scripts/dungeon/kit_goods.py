"""Goods for the maze's shops as real 3D objects: PET bottles, cans, snack bags, boxes, cup noodles,
rice balls, sandwiches, lunch boxes, drug store bottles and boxes, magazines, books, bread, flowers,
cups, bowls and clothes. Many small things go into one mesh per shelf (one material: the packaging
atlas of goods_gfx.py), so a shop full of goods stays a handful of objects.

Also the furniture they stand on: gondola shelves with boards and price rails, a wall cooler with
glass doors and lit shelves, an open cooler, a magazine rack, bread display shelves.
"""
import math

import bpy
import numpy as np

import build_scene as B
import goods_gfx as G
import kit as K
from kit import put, bx

DOZ = 0.004       # gap between neighbouring goods


class Batch:
    """Faces with per-corner UVs collected into one mesh."""
    def __init__(self):
        self.V, self.F, self.U = [], [], []

    def face(self, pts, uvs):
        base = len(self.V)
        self.V.extend(tuple(p) for p in pts)
        self.F.append(tuple(range(base, base + len(pts))))
        self.U.extend(uvs)

    def quad_uv(self, pts, r):
        u0, v0, u1, v1 = r
        self.face(pts, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    def box(self, lo, hi, front_uv, other_uv=None, front="+y", top_uv=None):
        """Box lo..hi; the face looking toward `front` gets front_uv (bottom left at u0, v0)."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        o = other_uv or front_uv
        faces = {
            "+y": [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
            "-y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            "+x": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            "-x": [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
            "+z": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            "-z": [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)],
        }
        for k, pts in faces.items():
            r = front_uv if k == front else (top_uv if (k == "+z" and top_uv) else o)
            self.quad_uv(pts, r)

    def cyl(self, c, r, h, side_uv, top_uv, bottom_uv=None, r_top=None, segs=10, z_from=0.0, cap_top=True, cap_bottom=False):
        """Cylinder (or frustum) standing at c (bottom centre), side_uv wrapped once round."""
        cx, cy, cz = c
        rt = r if r_top is None else r_top
        u0, v0, u1, v1 = side_uv
        z0, z1 = cz + z_from, cz + z_from + h
        for i in range(segs):
            a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
            p = [(cx + r * math.cos(a0), cy + r * math.sin(a0), z0), (cx + r * math.cos(a1), cy + r * math.sin(a1), z0),
                 (cx + rt * math.cos(a1), cy + rt * math.sin(a1), z1), (cx + rt * math.cos(a0), cy + rt * math.sin(a0), z1)]
            ua, ub = u0 + (u1 - u0) * i / segs, u0 + (u1 - u0) * (i + 1) / segs
            self.face(p, [(ua, v0), (ub, v0), (ub, v1), (ua, v1)])
        for cap, zz, rr, rv in ((cap_top, z1, rt, top_uv), (cap_bottom, z0, r, bottom_uv or top_uv)):
            if not cap or rv is None or rr <= 0:
                continue
            tu0, tv0, tu1, tv1 = rv
            mu, mv, ru, rvv = (tu0 + tu1) / 2, (tv0 + tv1) / 2, (tu1 - tu0) / 2, (tv1 - tv0) / 2
            pts, uvs = [], []
            rng = range(segs) if zz == z1 else range(segs - 1, -1, -1)
            for i in rng:
                a = 2 * math.pi * i / segs
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), zz))
                uvs.append((mu + ru * math.cos(a), mv + rvv * math.sin(a)))
            self.face(pts, uvs)

    def ellipsoid(self, c, s, uvr, segs=7, rings=4):
        cx, cy, cz = c
        u0, v0, u1, v1 = uvr
        for j in range(rings):
            t0, t1 = math.pi * j / rings - math.pi / 2, math.pi * (j + 1) / rings - math.pi / 2
            for i in range(segs):
                a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs

                def P(t, a):
                    return (cx + s[0] * math.cos(t) * math.cos(a), cy + s[1] * math.cos(t) * math.sin(a), cz + s[2] * math.sin(t))
                p = [P(t0, a0), P(t0, a1), P(t1, a1), P(t1, a0)]
                ua, ub = u0 + (u1 - u0) * i / segs, u0 + (u1 - u0) * (i + 1) / segs
                va, vb = v0 + (v1 - v0) * j / rings, v0 + (v1 - v0) * (j + 1) / rings
                self.face(p, [(ua, va), (ub, va), (ub, vb), (ua, vb)])

    def prism_tri(self, base_c, w, h, d, front_uv, other_uv, front="+y"):
        """Triangular prism (a rice ball, a sandwich) standing with its triangle facing front."""
        x, y, z = base_c
        f, b = (y + d / 2, y - d / 2) if front == "+y" else (y - d / 2, y + d / 2)
        u0, v0, u1, v1 = front_uv
        tri = [(x - w / 2, z), (x + w / 2, z), (x, z + h)]
        tuv = [(u0 + (u1 - u0) * 0.06, v0 + (v1 - v0) * 0.08), (u0 + (u1 - u0) * 0.94, v0 + (v1 - v0) * 0.08), ((u0 + u1) / 2, v0 + (v1 - v0) * 0.92)]
        if front == "+y":
            self.face([(tri[1][0], f, tri[1][1]), (tri[0][0], f, tri[0][1]), (tri[2][0], f, tri[2][1])], [tuv[0], tuv[1], tuv[2]])
            self.face([(tri[0][0], b, tri[0][1]), (tri[1][0], b, tri[1][1]), (tri[2][0], b, tri[2][1])], tuv)
        else:
            self.face([(tri[0][0], f, tri[0][1]), (tri[1][0], f, tri[1][1]), (tri[2][0], f, tri[2][1])], tuv)
            self.face([(tri[1][0], b, tri[1][1]), (tri[0][0], b, tri[0][1]), (tri[2][0], b, tri[2][1])], tuv)
        ou0, ov0, ou1, ov1 = other_uv
        for (a, c) in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            pts = [(a[0], b, a[1]), (c[0], b, c[1]), (c[0], f, c[1]), (a[0], f, a[1])]
            if front == "+y":
                pts = pts[::-1]
            self.face(pts, [(ou0, ov0), (ou1, ov0), (ou1, ov1), (ou0, ov1)])

    def obj(self, name, mat, deco=True):
        if not self.F:
            return None
        me = bpy.data.meshes.new(f"{K.CUR}_{name}")
        me.from_pydata(self.V, [], self.F)
        me.update()
        lay = me.uv_layers.new(name="UVMap")
        lay.data.foreach_set("uv", np.array(self.U, np.float32).ravel())
        me.materials.append(mat)
        ob = bpy.data.objects.new(f"{K.CUR}_{name}", me)
        ob["vlm"] = 1          # lit per vertex in the app (no lightmap chart for every little item)
        bpy.context.scene.collection.objects.link(ob)
        return put(ob, deco=deco)


def cell(c, r):
    return G.cell_uv(c, r)


SW = G.swatch_uv
LIQ = ["clear_water", "tea_green", "tea_brown", "cola", "soda_yellow", "milk_tea", "orange_juice", "sports"]
CAP = ["cap_blue", "cap_green", "cap_orange", "cap_red", "cap_yellow", "cap_white", "cap_orange", "cap_blue"]


# ------------------------------------------------------------------ single goods (front toward +y)

def bottle(b, x, y, z, k):
    r, h = 0.033, 0.215
    b.cyl((x, y, z), r, 0.035, SW(LIQ[k]), None, segs=8)
    b.cyl((x, y, z), r, 0.10, cell(k, 0), None, segs=8, z_from=0.035, cap_top=False)
    b.cyl((x, y, z), r, 0.035, SW(LIQ[k]), None, segs=8, z_from=0.135, cap_top=False)
    b.cyl((x, y, z), r, 0.03, SW(LIQ[k]), None, r_top=0.013, segs=8, z_from=0.17, cap_top=False)
    b.cyl((x, y, z), 0.015, 0.022, SW(CAP[k]), SW(CAP[k]), segs=6, z_from=0.20)
    return h


def can(b, x, y, z, k):
    b.cyl((x, y, z), 0.033, 0.11, cell(k, 1), SW("can_top"), segs=8)
    return 0.11


def bag(b, x, y, z, k, w=0.17, d=0.06, h=0.24, lean=0.0):
    b.box((x - w / 2, y - d / 2, z), (x + w / 2, y + d / 2, z + h), cell(k, 2))
    return h


def boxgood(b, x, y, z, k, w=0.15, d=0.05, h=0.20):
    b.box((x - w / 2, y - d / 2, z), (x + w / 2, y + d / 2, z + h), cell(k, 3))
    return h


def cupnoodle(b, x, y, z, k):
    b.cyl((x, y, z), 0.042, 0.105, cell(k, 4), cell(4 + k, 4), r_top=0.05, segs=10)
    return 0.105


def onigiri(b, x, y, z, k):
    b.prism_tri((x, y, z), 0.105, 0.095, 0.04, cell(k, 5), SW("wrap_clear"))
    return 0.095


def sandwich(b, x, y, z, k):
    b.prism_tri((x, y, z), 0.13, 0.12, 0.06, cell(4 + k, 5), SW("wrap_clear"))
    return 0.12


def bento(b, x, y, z, k):
    b.box((x - 0.10, y - 0.075, z), (x + 0.10, y + 0.075, z + 0.05), SW("plastic_tray"), front="+y", top_uv=cell(6 + k, 5))
    return 0.05


def drugbottle(b, x, y, z, k):
    b.box((x - 0.04, y - 0.025, z), (x + 0.04, y + 0.025, z + 0.19), cell(k, 6))
    b.cyl((x, y, z + 0.19), 0.012, 0.03, SW("shampoo_pump"), SW("shampoo_pump"), segs=6)
    return 0.22


def drugbox(b, x, y, z, k):
    b.box((x - 0.06, y - 0.025, z), (x + 0.06, y + 0.025, z + 0.13), cell(k, 6))
    return 0.13


def magazine(b, x, y, z, k, lean=0.25):
    w, h, t = 0.21, 0.28, 0.008
    # leaning back against the rack: bottom edge at y, top edge pushed back
    dz, dy = h * math.cos(lean), h * math.sin(lean)
    bot_f, top_f = (x, y, z), (x, y - dy, z + dz)
    u0, v0, u1, v1 = cell(k, 7)
    p = [(x + w / 2, y, z), (x - w / 2, y, z), (x - w / 2, y - dy, z + dz), (x + w / 2, y - dy, z + dz)]
    b.face(p, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    back = [(q[0], q[1] - t, q[2]) for q in p]
    b.quad_uv(back[::-1], SW("magazine_edge"))
    del bot_f, top_f


# ------------------------------------------------------------------ rows of goods on a shelf board

KINDS = {
    "drinks": (bottle, 0.07, 8), "cans": (can, 0.07, 8), "snacks": (bag, 0.18, 8), "boxes": (boxgood, 0.16, 8),
    "cups": (cupnoodle, 0.105, 4), "onigiri": (onigiri, 0.115, 4), "sandwich": (sandwich, 0.14, 2),
    "bento": (bento, 0.21, 2), "drug_b": (drugbottle, 0.09, 8), "drug_box": (drugbox, 0.13, 8),
}


def row(b, kind, x0, x1, y_front, z, depth, rng, toward=1, facings=(2, 5), rows_back=None):
    """Fill a board from x0 to x1 at height z with goods of one kind, their fronts at y_front looking
    toward +y (toward=1) or -y, a few deep (rows_back), in facings of the same product."""
    fn, pitch, nvar = KINDS[kind]
    x = x0 + pitch / 2 + 0.01
    # only the front two rows show (and the second through gaps): no deeper stock is modelled
    back = rows_back or min(2, max(1, int(depth / (pitch * 0.6))))
    while x + pitch / 2 <= x1:
        k = int(rng.integers(nvar))
        n = int(rng.integers(facings[0], facings[1] + 1))
        for _ in range(n):
            if x + pitch / 2 > x1:
                break
            gone = rng.random() < 0.12          # a few items already sold: the front one missing
            for j in range(back):
                if j == 0 and gone:
                    continue
                yy = y_front - toward * (pitch * 0.45 + j * pitch * 0.75)
                if toward < 0:
                    yy = y_front + (pitch * 0.45 + j * pitch * 0.75)
                fn(b, x + rng.uniform(-0.004, 0.004), yy, z, k)
            x += pitch + DOZ


def turn(b, xc):
    """Turn a batch half round about the vertical line x = xc, y = 0 (goods made facing +y then face
    -y; turned, not mirrored, so labels still read the right way)."""
    b.V = [(2 * xc - x, -y, z) for (x, y, z) in b.V]
    return b


# ------------------------------------------------------------------ furniture

def gondola(p, x0, x1, yc, sides, kinds, rng, h=1.55, boards=(0.10, 0.48, 0.86, 1.24), depth=0.42):
    """Island shelf: a white back panel along x at yc, boards both sides, a price rail on every board
    with yellow tags, goods (one kind per board, listed per side) facing out."""
    m = K.M
    bx(f"{p}gp{x0:.1f}{yc:.1f}", (x0, yc - 0.025, 0.0), (x1, yc + 0.025, h), m["shelf_white"])
    bx(f"{p}gt{x0:.1f}{yc:.1f}", (x0, yc - depth, h - 0.03), (x1, yc + depth, h), m["shelf_white"], deco=True)
    for s in sides:
        sg = 1 if s == "+y" else -1
        ya, yb = sorted((yc + sg * 0.025, yc + sg * depth))
        bx(f"{p}gk{x0:.1f}{yc:.1f}{s}", (x0, ya, 0.0), (x1, yb, 0.10), m["clad_dark"])
        for e in (x0, x1):
            bx(f"{p}ge{e:.1f}{yc:.1f}{s}", (e - 0.02, ya, 0.0), (e + 0.02, yb, h), m["shelf_white"])
    batch = Batch()
    tags = Batch()
    for s, kl in zip(sides, kinds):
        sg = 1 if s == "+y" else -1
        ya, yb = sorted((yc + sg * 0.025, yc + sg * depth))
        y_front = yb if sg > 0 else ya
        for bi, zb in enumerate(boards):
            if bi > 0:
                bx(f"{p}gb{x0:.1f}{yc:.1f}{s}{bi}", (x0, ya, zb - 0.02), (x1, yb, zb), m["shelf_white"], deco=True)
            # price rail with tags
            ry0, ry1 = sorted((y_front, y_front + sg * 0.012))
            bx(f"{p}gr{x0:.1f}{yc:.1f}{s}{bi}", (x0, ry0, zb - 0.045), (x1, ry1, zb), m["price_rail"], deco=True)
            for tx in np.arange(x0 + 0.12, x1 - 0.05, 0.28):
                yy = y_front + sg * 0.013
                tags.quad_uv([(tx + 0.04, yy, zb - 0.04), (tx - 0.04, yy, zb - 0.04), (tx - 0.04, yy, zb - 0.005), (tx + 0.04, yy, zb - 0.005)]
                             if sg > 0 else [(tx - 0.04, yy, zb - 0.04), (tx + 0.04, yy, zb - 0.04), (tx + 0.04, yy, zb - 0.005), (tx - 0.04, yy, zb - 0.005)],
                             SW("price_yellow"))
            kind = kl[bi % len(kl)]
            sub = Batch()
            row(sub, kind, x0 + 0.03, x1 - 0.03, 0.0, zb, depth - 0.06, rng)
            if sg < 0:
                turn(sub, (x0 + x1) / 2)
            for (vx, vy, vz) in sub.V:
                batch.V.append((vx, vy + y_front - sg * 0.015, vz))
            off = len(batch.V) - len(sub.V)
            batch.F.extend(tuple(i + off for i in f) for f in sub.F)
            batch.U.extend(sub.U)
    batch.obj(f"{p}goods{x0:.1f}{yc:.1f}", m["goods"])
    tags.obj(f"{p}tags{x0:.1f}{yc:.1f}", m["goods"])


def wall_cooler(p, x0, x1, y_back, rng, h=2.05, depth=0.7, kinds=("drinks", "drinks", "cans", "drinks", "cans")):
    """Walk-in style drinks cooler along a back wall: frame, lit shelves of bottles and cans behind
    glass doors with handles."""
    m = K.M
    yf = y_back + depth
    bx(f"{p}cb", (x0, y_back, 0.0), (x1, y_back + 0.05, h), m["shelf_white"])
    bx(f"{p}ct", (x0, y_back, h), (x1, yf, h + 0.25), m["clad_dark"])
    bx(f"{p}cbase", (x0, y_back, 0.0), (x1, yf, 0.12), m["clad_dark"])
    nd = max(1, int(round((x1 - x0) / 0.75)))
    dw = (x1 - x0) / nd
    batch = Batch()
    for i in range(nd + 1):
        xx = x0 + i * dw
        bx(f"{p}cm{i}", (xx - 0.025, yf - 0.06, 0.12), (xx + 0.025, yf, h), m["steel_brushed"])
    for i in range(nd):
        xa = x0 + i * dw
        # light strip down the mullion, glass door, handle
        put(B.image_quad(f"{K.CUR}_{p}cl{i}", (xa + 0.03, yf - 0.065, (h + 0.12) / 2), (0, 1, 0), (0, 0, 1), 0.02, h - 0.2, m["fridge"]), deco=True)
        bx(f"{p}cg{i}", (xa + 0.025, yf - 0.03, 0.12), (xa + dw - 0.025, yf - 0.015, h), m["shop_glass"], deco=True)
        bx(f"{p}ch{i}", (xa + dw - 0.09, yf - 0.012, 0.75), (xa + dw - 0.07, yf + 0.02, 1.45), m["steel"], deco=True)
        for j, zb in enumerate(np.linspace(0.16, h - 0.38, 5)):
            bx(f"{p}cs{i}{j}", (xa + 0.03, y_back + 0.05, zb - 0.015), (xa + dw - 0.03, yf - 0.08, zb), m["shelf_white"], deco=True)
            row(batch, kinds[j % len(kinds)], xa + 0.04, xa + dw - 0.04, yf - 0.10, zb, depth - 0.2, rng, facings=(1, 3), rows_back=2)
    batch.obj(f"{p}cgoods", m["goods"])
    ob = B.area_light(f"{K.CUR}_{p}cA", ((x0 + x1) / 2, yf - 0.2, h - 0.05), x1 - x0 - 0.2, 0.3, 260.0, (0.95, 0.97, 1.0), camera=False)
    put(ob, deco=True)


def open_cooler(p, x0, x1, yc, rng):
    """Low open cooler (rice balls, sandwiches, lunch boxes on stepped shelves), shoppers' side +y."""
    m = K.M
    bx(f"{p}ob", (x0, yc - 0.45, 0.0), (x1, yc + 0.25, 0.75), m["shelf_white"])
    bx(f"{p}oback", (x0, yc - 0.45, 0.75), (x1, yc - 0.35, 1.45), m["shelf_white"])
    bx(f"{p}otop", (x0, yc - 0.45, 1.45), (x1, yc + 0.1, 1.5), m["clad_dark"], deco=True)
    batch = Batch()
    for j, (zb, yy, kind) in enumerate(((0.75, yc + 0.22, "bento"), (0.95, yc + 0.02, "onigiri"), (1.15, yc - 0.15, "sandwich"))):
        bx(f"{p}ost{j}", (x0, yc - 0.45, zb - 0.02), (x1, yy, zb), m["shelf_white"], deco=True)
        row(batch, kind, x0 + 0.03, x1 - 0.03, yy - 0.01, zb, 0.2, rng, facings=(2, 4), rows_back=2 if kind != "bento" else 1)
    batch.obj(f"{p}ogoods", m["goods"])
    put(B.image_quad(f"{K.CUR}_{p}olt", ((x0 + x1) / 2, yc - 0.2, 1.445), (1, 0, 0), (0, -1, 0), x1 - x0 - 0.1, 0.1, m["fridge"]), deco=True)


def magazine_rack(p, x0, x1, y_wall, rng):
    """Low rack along the front glass, magazines leaning in two tiers (they face -y: into the shop)."""
    m = K.M
    bx(f"{p}mr", (x0, y_wall - 0.35, 0.0), (x1, y_wall, 0.7), m["shelf_white"])
    batch = Batch()
    for t, (zb, yy) in enumerate(((0.7, y_wall - 0.33), (0.95, y_wall - 0.1))):
        if t == 1:
            bx(f"{p}mr2", (x0, y_wall - 0.12, 0.7), (x1, y_wall, 0.95), m["shelf_white"])
        x = x0 + 0.12
        while x < x1 - 0.12:
            k = int(rng.integers(4))
            sub = Batch()
            magazine(sub, x, 0.0, 0.0, k)
            turn(sub, x)
            for (vx, vy, vz) in sub.V:
                batch.V.append((vx, vy + yy, vz + zb))
            off = len(batch.V) - len(sub.V)
            batch.F.extend(tuple(i + off for i in f) for f in sub.F)
            batch.U.extend(sub.U)
            x += 0.23
    batch.obj(f"{p}mags", m["goods"])


def bookcase(p, x0, x1, y_wall, rng, h=2.1, shelves=6, toward=1):
    """Wall bookcase: wooden carcass, books of varied height and thickness spine out on every shelf,
    now and then a few lying flat or leaning."""
    m = K.M
    d = 0.32
    ya, yb = sorted((y_wall, y_wall + toward * d))
    bx(f"{p}bkb", (x0, ya, 0.0), (x1, yb, 0.08), m["wood"])
    bx(f"{p}bkk", (x0, ya if toward > 0 else yb - 0.02, 0.0), (x1, ya + 0.02 if toward > 0 else yb, h), m["wood"])
    for e in (x0, x1):
        bx(f"{p}bke{e:.1f}", (e - 0.02, ya, 0.0), (e + 0.02, yb, h), m["wood"])
    batch = Batch()
    yf = yb if toward > 0 else ya
    for s in range(shelves):
        zb = 0.08 + s * (h - 0.1) / shelves
        if s:
            bx(f"{p}bks{s}", (x0, ya, zb - 0.02), (x1, yb, zb), m["wood"], deco=True)
        x = x0 + 0.03
        top = zb + (h - 0.1) / shelves - 0.05
        while x < x1 - 0.06:
            t = rng.uniform(0.018, 0.045)
            hh = min(top - zb, rng.uniform(0.18, 0.27))
            i = int(rng.integers(64))
            u0, v0, u1, v1 = G.spine_uv(i)
            yfront = yf - toward * rng.uniform(0.0, 0.02)
            yback = yfront - toward * rng.uniform(0.15, 0.22)
            y0_, y1_ = sorted((yfront, yback))
            # spine (front) with the title; covers and page edges
            batch.box((x, y0_, zb), (x + t, y1_, zb + hh), (u0, v0, u1, v1), other_uv=PAGES, front="+y" if toward > 0 else "-y")
            x += t + 0.002
            if rng.random() < 0.04:
                x += rng.uniform(0.05, 0.12)
    batch.obj(f"{p}books{x0:.1f}", m["books"])


PAGES = (0.50, 0.12, 0.52, 0.14)        # a patch of page colour in books.png (below the covers)


def book_table(p, x0, x1, y0, y1, rng):
    """Low table with books lying in stacks, covers up."""
    m = K.M
    bx(f"{p}bt{x0:.1f}", (x0, y0, 0.0), (x1, y1, 0.72), m["wood"])
    batch = Batch()
    for i, x in enumerate(np.arange(x0 + 0.2, x1 - 0.1, 0.3)):
        for j, y in enumerate(np.arange(y0 + 0.2, y1 - 0.1, 0.32)):
            n = int(rng.integers(2, 6))
            k = int(rng.integers(8))
            for q in range(n):
                z = 0.72 + q * 0.025
                batch.box((x - 0.1, y - 0.14, z), (x + 0.1, y + 0.14, z + 0.024), PAGES, other_uv=PAGES, front="+y", top_uv=G.cover_uv(k))
    batch.obj(f"{p}btb{x0:.1f}", m["books"])


def bread_shelf(p, x0, x1, y_wall, rng, toward=1, tiers=(0.75, 1.1, 1.45)):
    """Bread on trays on a tiered wooden display: round buns, loaves, baguettes, croissant-like rolls."""
    m = K.M
    d = 0.55
    ya, yb = sorted((y_wall, y_wall + toward * d))
    bx(f"{p}brb", (x0, ya, 0.0), (x1, yb, tiers[0] - 0.02), m["wood"])
    batch = Batch()
    for t, zb in enumerate(tiers):
        yy0, yy1 = sorted((y_wall, y_wall + toward * (d - t * 0.12)))
        if t:
            bx(f"{p}brt{t}", (x0, yy0, zb - 0.03), (x1, yy1, zb), m["wood"], deco=True)
            bx(f"{p}brp{t}", (x0 + 0.02, (yy0 + yy1) / 2 - 0.02, tiers[t - 1]), (x0 + 0.06, (yy0 + yy1) / 2 + 0.02, zb), m["wood"], deco=True)
            bx(f"{p}brq{t}", (x1 - 0.06, (yy0 + yy1) / 2 - 0.02, tiers[t - 1]), (x1 - 0.02, (yy0 + yy1) / 2 + 0.02, zb), m["wood"], deco=True)
        x = x0 + 0.05
        while x < x1 - 0.45:
            tw = 0.42
            batch.box((x, yy0 + 0.03, zb), (x + tw, yy1 - 0.03, zb + 0.012), SW("plastic_tray"), front="+z")
            kind = int(rng.integers(4))
            crust = [cell(5, 7), cell(6, 7), cell(7, 7)][int(rng.integers(3))]
            cy = (yy0 + yy1) / 2
            if kind == 0:          # round buns, heaped
                for i in range(3):
                    for j in range(3):
                        batch.ellipsoid((x + 0.08 + i * 0.13, cy - 0.12 + j * 0.12, zb + 0.045), (0.065, 0.065, 0.05), crust)
                for i in range(2):
                    batch.ellipsoid((x + 0.14 + i * 0.13, cy, zb + 0.12), (0.065, 0.065, 0.05), crust)
            elif kind == 1:        # loaves
                for i in range(2):
                    for j in range(2):
                        batch.ellipsoid((x + 0.11 + i * 0.2, cy - 0.08 + j * 0.16, zb + 0.07), (0.1, 0.07, 0.07), crust)
            elif kind == 2:        # baguettes
                for j in range(5):
                    batch.ellipsoid((x + tw / 2, cy - 0.14 + j * 0.07, zb + 0.035), (0.2, 0.035, 0.033), crust)
            else:                  # rolls
                for i in range(4):
                    for j in range(3):
                        batch.ellipsoid((x + 0.06 + i * 0.1, cy - 0.1 + j * 0.1, zb + 0.032), (0.05, 0.035, 0.032), crust)
            # price card
            yc_ = yy1 - 0.02 if toward > 0 else yy0 + 0.02
            card = [(x + tw - 0.02, yc_, zb + 0.012), (x + tw - 0.12, yc_, zb + 0.012), (x + tw - 0.12, yc_, zb + 0.07), (x + tw - 0.02, yc_, zb + 0.07)]
            batch.quad_uv(card if toward > 0 else card[::-1], SW("paper"))
            x += tw + 0.04
    batch.obj(f"{p}bread{x0:.1f}", m["goods"])


def drug_gondola(p, x0, x1, yc, rng):
    gondola(p, x0, x1, yc, ("+y", "-y"), (("drug_box", "drug_b", "drug_box", "drug_b"), ("drug_b", "drug_box", "boxes", "drug_box")), rng, h=1.6)


def flowers(p, spots, rng, z0=0.0):
    """Buckets of cut flowers: a dense bunch per bucket of one or two colours, stems, leaves."""
    m = K.M
    batch = Batch()
    cols = ["petal_red", "petal_yellow", "petal_pink", "petal_white", "petal_purple"]
    for k, (x0, y0) in enumerate(spots):
        batch.cyl((x0, y0, z0), 0.17, 0.36, SW("steel"), SW("ice"), r_top=0.19, segs=14)
        c1, c2 = cols[(k + int(rng.integers(5))) % 5], cols[(k * 3 + 1) % 5]
        for j in range(26):
            a = rng.uniform(0, 2 * math.pi)
            r = 0.15 * math.sqrt(rng.uniform(0, 1))
            hx, hy = x0 + math.cos(a) * r, y0 + math.sin(a) * r
            hz = z0 + 0.62 + rng.uniform(0, 0.2) - r * 0.6
            batch.cyl((hx, hy, z0 + 0.34), 0.005, hz - z0 - 0.34, SW("stem"), None, segs=4, cap_top=False)
            rr = rng.uniform(0.04, 0.06)
            batch.ellipsoid((hx, hy, hz), (rr, rr, rr * 0.7), SW(c1 if j % 3 else c2), segs=6, rings=3)
            if j % 2 == 0:
                la = rng.uniform(0, 2 * math.pi)
                lx, ly = hx + math.cos(la) * 0.08, hy + math.sin(la) * 0.08
                leaf = [(hx, hy, hz - 0.2), (lx, ly, hz - 0.16), (lx, ly, hz - 0.06), (hx, hy, hz - 0.1)]
                batch.quad_uv(leaf, SW("leaf"))
                batch.quad_uv(leaf[::-1], SW("leaf"))
    batch.obj(f"{p}flowers", m["goods"])


def clothes_rack(p, x0, x1, y, rng):
    """Garment rail with shirts and jackets on hangers (body, sleeves, collar) in a few colours."""
    m = K.M
    bx(f"{p}rk{x0:.1f}", (x0, y - 0.015, 1.5), (x1, y + 0.015, 1.53), m["steel"], deco=True)
    for e in (x0, x1):
        bx(f"{p}rkp{e:.1f}", (e - 0.015, y - 0.015, 0.0), (e + 0.015, y + 0.015, 1.53), m["steel"])
        bx(f"{p}rkf{e:.1f}", (e - 0.02, y - 0.3, 0.0), (e + 0.02, y + 0.3, 0.03), m["steel"])
    batch = Batch()
    cols = ["fabric_navy", "fabric_grey", "fabric_beige", "fabric_white", "fabric_black", "fabric_green", "fabric_red", "denim"]
    x = x0 + 0.08
    while x < x1 - 0.06:
        c = SW(cols[int(rng.integers(len(cols)))])
        t = rng.uniform(0.025, 0.04)
        top = 1.47
        ln = rng.uniform(0.62, 0.9)
        batch.box((x, y - 0.23, top - ln), (x + t, y + 0.23, top), c, front="+x")
        batch.box((x, y - 0.29, top - 0.42), (x + t * 0.8, y - 0.23, top - 0.04), c, front="+x")
        batch.box((x, y + 0.23, top - 0.42), (x + t * 0.8, y + 0.29, top - 0.04), c, front="+x")
        batch.box((x + t * 0.3, y - 0.005, top), (x + t * 0.7, y + 0.005, 1.53), SW("steel"), front="+x")
        x += t + rng.uniform(0.02, 0.05)
    batch.obj(f"{p}clothes{x0:.1f}", m["goods"])


def folded_stacks(p, x0, x1, y0, y1, rng):
    m = K.M
    bx(f"{p}ft{x0:.1f}", (x0, y0, 0.0), (x1, y1, 0.75), m["wood_light"])
    batch = Batch()
    cols = ["fabric_navy", "fabric_grey", "fabric_beige", "fabric_white", "fabric_black", "denim"]
    for x in np.arange(x0 + 0.2, x1 - 0.15, 0.36):
        for y in np.arange(y0 + 0.2, y1 - 0.15, 0.4):
            c = SW(cols[int(rng.integers(len(cols)))])
            for q in range(int(rng.integers(3, 7))):
                z = 0.75 + q * 0.035
                batch.box((x - 0.15, y - 0.17, z), (x + 0.15, y + 0.17, z + 0.033), c, front="+y")
    batch.obj(f"{p}fold{x0:.1f}", m["goods"])


def cafe_items(p, tables, counter_y, rng):
    """Cups on saucers on the tables, a cake case on the counter with cakes, cups by the machine."""
    m = K.M
    batch = Batch()
    for (x, y) in tables:
        for k in range(int(rng.integers(1, 3))):
            a = rng.uniform(0, 2 * math.pi)
            cx, cy = x + math.cos(a) * 0.15, y + math.sin(a) * 0.15
            batch.cyl((cx, cy, 0.72), 0.075, 0.008, SW("cup_white"), SW("cup_white"), segs=12)
            batch.cyl((cx, cy, 0.728), 0.04, 0.07, SW("cup_white"), SW("coffee"), r_top=0.045, segs=12)
    batch.obj(f"{p}cups", m["goods"])
    # cake case: glass box on the counter with three tiers of cake slices
    cx0, cx1 = 1.3, 3.1
    bx(f"{p}cake_b", (cx0, counter_y - 0.3, 1.0), (cx1, counter_y + 0.3, 1.05), m["steel_brushed"], deco=True)
    bx(f"{p}cake_g", (cx0, counter_y - 0.3, 1.05), (cx1, counter_y + 0.3, 1.45), m["shop_glass"], deco=True)
    put(B.image_quad(f"{K.CUR}_{p}cake_l", ((cx0 + cx1) / 2, counter_y, 1.445), (1, 0, 0), (0, -1, 0), cx1 - cx0 - 0.1, 0.4, m["fridge"]), deco=True)
    cb = Batch()
    # whole round cakes on paper doilies: sponge sides, cream or chocolate tops, a ring of strawberries
    for i, x in enumerate(np.arange(cx0 + 0.2, cx1 - 0.15, 0.36)):
        top = ["cream", "chocolate", "cream", "strawberry", "cream"][i % 5]
        cb.cyl((x, counter_y, 1.05), 0.13, 0.004, SW("paper"), SW("paper"), segs=16)
        cb.cyl((x, counter_y, 1.054), 0.11, 0.075, SW("bread_light") if top != "chocolate" else SW("chocolate"), SW(top), segs=18)
        for k in range(8):
            a_ = 2 * math.pi * k / 8
            cb.ellipsoid((x + math.cos(a_) * 0.08, counter_y + math.sin(a_) * 0.08, 1.141), (0.016, 0.016, 0.014),
                         SW("strawberry" if top != "strawberry" else "cream"), segs=6, rings=3)
    cb.obj(f"{p}cakes", m["goods"])


def ramen_items(p, x0, x1, y_counter, rng):
    m = K.M
    batch = Batch()
    for x in np.arange(x0 + 0.4, x1 - 0.3, 0.75):
        if rng.random() < 0.5:
            batch.cyl((x, y_counter, 1.0), 0.05, 0.07, SW("bowl_red"), SW("broth"), r_top=0.1, segs=14)
        batch.cyl((x + 0.2, y_counter + 0.05, 1.0), 0.03, 0.09, SW("ice"), SW("ice"), segs=8)
        batch.box((x - 0.2, y_counter - 0.05, 1.0), (x - 0.12, y_counter + 0.05, 1.12), SW("black"), front="+y")
    batch.obj(f"{p}ramen_items", m["goods"])


def materials(M):
    """Goods atlas and books atlas materials (called from kit.materials)."""
    gp = G.build_goods()
    bp = G.build_books()
    M["goods"] = B.printed("goods_atlas", gp, rough=0.35, coat=0.3, emission=0.0)
    M["books"] = B.printed("books_atlas", bp, rough=0.6, coat=0.0, emission=0.0)
    M["price_rail"] = B.plain("price_rail", (0.85, 0.86, 0.86), 0.3, coat=0.3)
