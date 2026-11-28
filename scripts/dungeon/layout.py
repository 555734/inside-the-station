"""Lays the modules of the maze out on the 16 m grid and draws the map (plain Python, no Blender).

  python3 scripts/dungeon/layout.py OUT_DIR [--seed 7] [--cells 110]

Grows from the first station outwards: an open port is taken at random and a module is fitted to it
(any module, turned any way) only if every edge of its cells agrees with what is already there: a
port meets a port, a wall meets a wall or an empty cell. So no opening ever runs into a wall and no
wall stands in a passage. When the maze has its size, every port still open gets the module that
closes it (an exit stair up to the street, or the corridor piece joining the ports that meet there).
Writes layout.json (instances, spawn points, exit numbers) and map.png.
"""
import argparse
import json
import math
import os
import random

C = 16.0
HC = C / 2
SIDES = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
ROT = {"N": "W", "W": "S", "S": "E", "E": "N"}         # a quarter turn anticlockwise
ONE = [(0.0, 0.0)]
TWO = [(-HC, -HC), (HC, -HC), (-HC, HC), (HC, HC)]
NS = [(0.0, HC, "N"), (0.0, -HC, "S")]
CAT = {
    "I": (ONE, NS), "I2": (ONE, NS), "SHOP_A": (ONE, NS), "SHOP_B": (ONE, NS), "SHOP_C": (ONE, NS),
    "L": (ONE, [(0.0, -HC, "S"), (HC, 0.0, "E")]),
    "T": (ONE, [(0.0, HC, "N"), (0.0, -HC, "S"), (HC, 0.0, "E")]),
    "X": (ONE, [(0.0, HC, "N"), (0.0, -HC, "S"), (HC, 0.0, "E"), (-HC, 0.0, "W")]),
    "EXIT": (ONE, [(0.0, -HC, "S")]),
    "PLAZA": (TWO, [(-8.0, C, "N"), (C, 8.0, "E"), (8.0, -C, "S"), (-C, -8.0, "W")]),
    "STATION": ([(-C, -HC), (0.0, -HC), (C, -HC), (-C, HC), (0.0, HC), (C, HC)],
                [(-C, C, "N"), (0.0, C, "N"), (C, C, "N"), (-1.5 * C, HC, "W"), (1.5 * C, HC, "E")]),
}
WEIGHT = {"I": 3.0, "I2": 1.0, "SHOP_A": 1.3, "SHOP_B": 1.3, "SHOP_C": 1.3, "L": 1.3, "T": 1.1, "X": 0.5,
          "PLAZA": 0.35, "STATION": 0.4, "EXIT": 0.12}
LIMIT = {"PLAZA": 3, "STATION": 2}
PLATFORM_HALF = (96.0, 24.0)     # the platform level under a station (x, y half extents, track tunnels incl.)


def rot(p, r):
    x, y = p
    for _ in range(r % 4):
        x, y = -y, x
    return x, y


def rot_side(s, r):
    for _ in range(r % 4):
        s = ROT[s]
    return s


def cell_of(x, y):
    return (int(round(x / C)), int(round(y / C)))


class Maze:
    def __init__(self, seed, bounds):
        self.rng = random.Random(seed)
        self.occ = {}            # cell -> instance index
        self.inst = []           # {"module", "pos", "rot", "cells", "ports": [(edge key, side)]}
        self.bounds = bounds     # (i0, j0, i1, j1) inclusive
        self.count = {}
        self.platforms = []      # rectangles (x0, y0, x1, y1) of the platform levels

    def placed(self, m, P, r):
        cells, ports = CAT[m]
        wc = [cell_of(P[0] + rot(c, r)[0], P[1] + rot(c, r)[1]) for c in cells]
        wp = []
        for (x, y, s) in ports:
            px, py = rot((x, y), r)
            wp.append(((round(P[0] + px, 3), round(P[1] + py, 3)), rot_side(s, r)))
        return wc, wp

    def port_at(self, cell, side):
        """Is there a port on this edge of the (occupied) cell?"""
        k = self.occ.get(cell)
        if k is None:
            return None
        cx, cy = cell[0] * C, cell[1] * C
        dx, dy = SIDES[side]
        key = (round(cx + dx * HC, 3), round(cy + dy * HC, 3))
        return any(pk == key and ps == side for pk, ps in self.inst[k]["ports"])

    def fits(self, m, P, r):
        wc, wp = self.placed(m, P, r)
        i0, j0, i1, j1 = self.bounds
        mine = set(wc)
        if any(c in self.occ or not (i0 <= c[0] <= i1 and j0 <= c[1] <= j1) for c in wc):
            return False
        pset = {(k, s) for k, s in wp}
        for c in wc:
            for s, (dx, dy) in SIDES.items():
                n = (c[0] + dx, c[1] + dy)
                if n in mine:
                    continue
                key = (round(c[0] * C + dx * HC, 3), round(c[1] * C + dy * HC, 3))
                ours = (key, s) in pset
                if n in self.occ:
                    if bool(self.port_at(n, OPP[s])) != ours:
                        return False
                elif ours and not (i0 <= n[0] <= i1 and j0 <= n[1] <= j1):
                    return False
        if m == "STATION":
            x0, y0 = P[0] - PLATFORM_HALF[0], P[1] - PLATFORM_HALF[1]
            x1, y1 = P[0] + PLATFORM_HALF[0], P[1] + PLATFORM_HALF[1]
            if r % 2:
                x0, y0, x1, y1 = P[0] - PLATFORM_HALF[1], P[1] - PLATFORM_HALF[0], P[0] + PLATFORM_HALF[1], P[1] + PLATFORM_HALF[0]
            if any(not (x1 < a or b < x0 or y1 < c or d < y0) for a, c, b, d in self.platforms):
                return False
        return True

    def place(self, m, P, r):
        wc, wp = self.placed(m, P, r)
        k = len(self.inst)
        self.inst.append({"module": m, "pos": [round(P[0], 3), round(P[1], 3)], "rot": r, "cells": wc, "ports": wp})
        for c in wc:
            self.occ[c] = k
        self.count[m] = self.count.get(m, 0) + 1
        if m == "STATION":
            hx, hy = PLATFORM_HALF if r % 2 == 0 else PLATFORM_HALF[::-1]
            self.platforms.append((P[0] - hx, P[1] - hy, P[0] + hx, P[1] + hy))
        return k

    def open_ports(self):
        out = []
        for k, it in enumerate(self.inst):
            for key, s in it["ports"]:
                dx, dy = SIDES[s]
                n = cell_of(key[0] + dx * HC, key[1] + dy * HC)
                if n not in self.occ:
                    out.append((key, s, n))
        return out

    def candidates(self, key, side, mods):
        """Placements of these modules with one of their ports on edge key, facing side's opposite."""
        out = []
        need = OPP[side]
        for m in mods:
            cells, ports = CAT[m]
            for r in range(4):
                for (x, y, s) in ports:
                    if rot_side(s, r) != need:
                        continue
                    px, py = rot((x, y), r)
                    P = (key[0] - px, key[1] - py)
                    if self.fits(m, P, r):
                        out.append((m, P, r))
        return out

    def grow(self, target):
        while sum(len(i["cells"]) for i in self.inst) < target:
            op = self.open_ports()
            if not op:
                break
            key, side, n = self.rng.choice(op)
            mods = [m for m in WEIGHT if self.count.get(m, 0) < LIMIT.get(m, 10 ** 9)]
            if "STATION" in mods and self.dist_to_station(n) < 7:
                mods.remove("STATION")
            cands = self.candidates(key, side, mods)
            if not cands:
                continue
            w = [WEIGHT[c[0]] for c in cands]
            m, P, r = self.rng.choices(cands, weights=w)[0]
            self.place(m, P, r)

    def dist_to_station(self, n):
        d = 1e9
        for it in self.inst:
            if it["module"] == "STATION":
                d = min(d, min(abs(n[0] - c[0]) + abs(n[1] - c[1]) for c in it["cells"]))
        return d

    def close(self):
        """Every open port gets the one-cell module whose ports are exactly the ports facing that cell."""
        guard = 0
        while True:
            op = self.open_ports()
            if not op or guard > 10000:
                break
            guard += 1
            key, side, n = op[0]
            for m in ("EXIT", "I", "L", "T", "X"):
                c = self.candidates(key, side, [m])
                if c:
                    self.place(*c[0])
                    break
            else:
                raise RuntimeError(f"cannot close port at {key}")


def build(seed, cells, bounds=(-10, -8, 10, 8)):
    for attempt in range(50):
        mz = Maze(seed + attempt * 1000, bounds)
        mz.place("STATION", (0.0, HC), 0)
        mz.grow(cells)
        if mz.count.get("STATION", 0) >= 2 and mz.count.get("PLAZA", 0) >= 1:
            mz.close()
            return mz
    raise RuntimeError("no layout with two stations")


def draw_map(mz, path, exits):
    from PIL import Image, ImageDraw
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from graphics import F
    xs = [c[0] for it in mz.inst for c in it["cells"]]
    ys = [c[1] for it in mz.inst for c in it["cells"]]
    i0, i1, j0, j1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    S = 56
    W, H = (i1 - i0 + 1) * S, (j1 - j0 + 1) * S + 120
    im = Image.new("RGB", (W, H), (245, 244, 240))
    d = ImageDraw.Draw(im)

    def px(x, y):          # metres -> pixels
        return ((x / C - i0 + 0.5) * S, (j1 - y / C + 0.5) * S)
    col = {"I": (210, 210, 205), "I2": (200, 205, 215), "L": (210, 210, 205), "T": (210, 210, 205), "X": (210, 210, 205),
           "SHOP_A": (250, 205, 140), "SHOP_B": (250, 190, 160), "SHOP_C": (245, 215, 160),
           "EXIT": (255, 225, 60), "PLAZA": (170, 215, 180), "STATION": (120, 200, 170)}
    for it in mz.inst:
        for c in it["cells"]:
            x, y = px(c[0] * C, c[1] * C)
            d.rectangle([x - S / 2 + 1, y - S / 2 + 1, x + S / 2 - 1, y + S / 2 - 1], fill=(232, 231, 226))
    # walkable parts: corridors as bars between the cell centre and its ports, halls as squares
    for it in mz.inst:
        m = it["module"]
        c0 = col[m]
        if m in ("PLAZA", "STATION"):
            x, y = px(*it["pos"])
            d.rectangle([x - S * 0.82, y - S * 0.82, x + S * 0.82, y + S * 0.82], fill=c0)
            for key, s in it["ports"]:
                a = px(*key)
                d.line([a, (x + (a[0] - x) * 0.6 if s in "EW" else a[0], y + (a[1] - y) * 0.6 if s in "NS" else a[1])], fill=c0, width=int(S * 0.38))
            label = "駅" if m == "STATION" else "広場"
            d.text((x, y), label, font=F("sansB", 26), fill=(20, 60, 40), anchor="mm")
            continue
        cx, cy = it["cells"][0]
        a = px(cx * C, cy * C)
        for key, s in it["ports"]:
            b = px(*key)
            d.line([a, b], fill=c0, width=int(S * 0.38))
        d.ellipse([a[0] - S * 0.19, a[1] - S * 0.19, a[0] + S * 0.19, a[1] + S * 0.19], fill=c0)
        if m.startswith("SHOP"):
            d.text(a, "店", font=F("sansB", 20), fill=(120, 60, 10), anchor="mm")
        if m == "EXIT":
            d.text(a, exits.get(id(it), ""), font=F("sansB", 16), fill=(30, 30, 30), anchor="mm")
    d.text((20, H - 90), "ひかり地下街 + 陽ヶ丘地下鉄  (1マス = 16 m)", font=F("sansB", 28), fill=(30, 30, 30))
    d.text((20, H - 45), "緑の大: 駅（改札・ホーム）  緑: 広場  橙: 店の並ぶ通り  黄: 地上への出口  灰: 通路",
           font=F("sansM", 20), fill=(60, 60, 60))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cells", type=int, default=110)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mz = build(a.seed, a.cells)
    exits, n = {}, 0
    for it in mz.inst:
        if it["module"] == "EXIT":
            n += 1
            exits[id(it)] = str(n)
    st = [it for it in mz.inst if it["module"] == "STATION"]
    out = {"cell": C, "instances": [{"module": it["module"], "pos": [it["pos"][0], it["pos"][1], 0.0], "rot": it["rot"],
                                     "exit": exits.get(id(it))} for it in mz.inst],
           "stations": [{"pos": s["pos"], "rot": s["rot"]} for s in st]}
    json.dump(out, open(os.path.join(a.out, "layout.json"), "w"), indent=1)
    draw_map(mz, os.path.join(a.out, "map.png"), exits)
    cnt = {}
    for it in mz.inst:
        cnt[it["module"]] = cnt.get(it["module"], 0) + 1
    cells = sum(len(i["cells"]) for i in mz.inst)
    print("instances", len(mz.inst), "cells", cells, "area", cells * C * C, "m2", cnt)


if __name__ == "__main__":
    main()
