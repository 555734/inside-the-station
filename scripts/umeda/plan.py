"""Vectorises the official Osaka Metro Umeda Station (Midosuji Line) premises map into 3D-ready polygons.

Source map: Osaka Metro, 避難確保・浸水防止計画 御堂筋線 梅田駅 (2018), page 9 「駅構内 避難経路図」.
Run scripts/umeda/fetch_map.sh first; it writes cache/umeda/map.png (the page raster, 1936x1369 px).

Upper drawing on the page = B1 concourse, lower drawing = B2 platform. Both share the same
horizontal scale and are aligned; the platform drawing sits PLAT_DY px lower on the page.
Scale is derived from the published platform widths (1番線側 12.0 m + 2番線側 12.5 m = 24.5 m)
and checks against the published effective lengths (197 m / 201 m).

Map left = north (至 中津), right = south (至 淀屋橋), top = east.
World frame (metres): +X = south, +Y = east, z = 0 at the B1 concourse floor.
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.path.join(ROOT, "cache", "umeda")

S = 0.187           # metres per map pixel
X0, Y0 = 1000, 650  # map pixel that becomes world (0, 0) on the B1 drawing
PLAT_DY = 493       # platform drawing offset on the page (px)

# ------------------------------------------------------------------ annotations (map px)
# Text labels, circled stair letters, title: erased before vectorising.
ERASE = [
    (0, 0, 1936, 395),                                    # title
    (470, 492, 545, 512), (445, 818, 505, 842),           # 2号, 3号出入口
    (910, 792, 975, 815), (1095, 858, 1150, 880),         # 9号, 8号出入口
    (1203, 805, 1262, 822), (1500, 812, 1565, 832),       # 7号, 18号出入口
    (1020, 452, 1072, 472), (1500, 480, 1565, 502),       # 8/6号, 10号出入口
    (1575, 420, 1650, 440), (1580, 440, 1680, 462),       # 阪急百貨店, 11号出入口
    (1655, 430, 1720, 470),                               # 12号出入口
    (1595, 840, 1680, 910), (1640, 845, 1700, 880),       # 7号出入口 中央通路(阪神), 店舗
    (355, 752, 420, 772),                                 # ヨドバシ梅田
    (100, 995, 200, 1015), (1630, 995, 1720, 1015),       # 至 中津 / 至 淀屋橋
    (1630, 1265, 1700, 1290),                             # 梅田駅
    # circled stair letters on the platform drawing
    (524, 1050, 545, 1066), (615, 1054, 636, 1072), (898, 1094, 916, 1110),
    (1223, 1050, 1244, 1068), (1440, 1071, 1462, 1092),
    (640, 1146, 662, 1166), (793, 1144, 814, 1164), (920, 1144, 942, 1164),
    (1162, 1150, 1184, 1170), (1460, 1180, 1482, 1200),
]

# Stairs between B1 and B2, in platform-drawing px.
# y0..y1 = stair width, x_top = edge at the concourse, x_bot = foot on the platform.
STAIRS = {
    "A": dict(y0=1065, y1=1094, x_top=505, x_bot=562),
    "B": dict(y0=1068, y1=1102, x_top=610, x_bot=668),
    "C": dict(y0=1064, y1=1094, x_top=872, x_bot=932),
    "D": dict(y0=1066, y1=1101, x_top=1180, x_bot=1250),
    "E": dict(y0=1060, y1=1104, x_top=1552, x_bot=1478),
    "F": dict(y0=1161, y1=1192, x_top=607, x_bot=666),
    "G": dict(y0=1163, y1=1193, x_top=777, x_bot=828),
    "H": dict(y0=1165, y1=1193, x_top=900, x_bot=960),
    "I": dict(y0=1163, y1=1195, x_top=1150, x_bot=1202),
    "J": dict(y0=1162, y1=1196, x_top=1555, x_bot=1480),
}

# Ticket gate lines on B1 (polyline, px). Each rung of the drawn "ladder" is one gate machine.
GATES = {
    "北改札": [(352, 558), (330, 600), (321, 640), (320, 745)],
    "中北西改札": [(1012, 628), (1012, 722)],
    "中北東改札": [(1112, 565), (1094, 610), (1094, 740)],
    "南改札": [(1606, 558), (1608, 740)],
}
GATE_ERASE = [(300, 550, 362, 750), (1000, 620, 1024, 730), (1084, 556, 1125, 745), (1596, 552, 1622, 748)]

# Exit stairs from B1 up to street level: centre, run length, width (px), up-direction (dx, dy).
EXITS = {
    "2": dict(c=(470, 484), run=52, w=26, up=(-0.18, -1)),
    "3": dict(c=(425, 822), run=62, w=46, up=(0.25, 1)),
    "8": dict(c=(1172, 850), run=62, w=48, up=(-0.35, 1)),
    "9": dict(c=(950, 765), run=40, w=36, up=(1, -0.3)),
    "6": dict(c=(984, 478), run=56, w=22, up=(-0.25, -1)),
    "10": dict(c=(1552, 506), run=24, w=14, up=(1, -0.5)),
    "11": dict(c=(1632, 505), run=58, w=34, up=(1, -0.45)),
    "18": dict(c=(1545, 800), run=86, w=18, up=(-1, 0)),
}
EXIT_ERASE = [(455, 452, 492, 512), (395, 785, 455, 860), (1135, 815, 1210, 885),
              (915, 745, 985, 785), (965, 440, 1000, 512), (995, 478, 1072, 498),
              (1535, 492, 1572, 518), (1580, 470, 1690, 540), (1495, 785, 1595, 812)]

# Inner corridor behind the north gate (barrel-vault ceiling, see the CC0 photo 大阪メトロ梅田駅北改札内通路)
B1_CORRIDOR = dict(x0=330, x1=1000, y0=663, y1=704)

# Concourse areas that are open voids down to B2 (vault of 2番線 south) - not walkable on B1.
VAULT = dict(x0=1222, x1=1470, y0_plat=1136, y1_plat=1222)

# Outer outline of the B1 drawing (px). Cuts the open ends where the station continues into
# neighbouring malls / exits; elsewhere it runs just outside the drawn outer walls.
B1_OUTLINE = [(292, 548), (425, 548), (425, 405), (500, 405), (500, 515), (535, 515), (535, 530),
              (1000, 530), (985, 500), (965, 445), (1000, 440), (1012, 500), (1075, 500), (1075, 505),
              (1555, 505), (1555, 470), (1700, 445),
              (1700, 835), (1610, 835), (1610, 815), (1490, 815), (1490, 770), (1265, 770), (1265, 890),
              (1140, 890), (1095, 820), (1040, 815), (1040, 775), (985, 775), (985, 790), (912, 790),
              (895, 752), (455, 752), (455, 860), (410, 866), (395, 765), (292, 765)]

SEEDS_B1 = [(700, 575), (1300, 555), (1150, 660), (500, 670), (1650, 650), (380, 640), (1050, 790), (1380, 760)]
SEEDS_B2 = [(700, 1075), (1300, 1080), (700, 1170), (1300, 1170), (1050, 1170)]
B1_ROI = (285, 400, 1700, 900)
B2_ROI = (440, 1040, 1700, 1215)


def load():
    img = cv2.imread(os.path.join(CACHE, "map.png")).astype(np.int16)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    red = (r - g > 50) & (r - b > 50)
    img[red] = 255
    black = img.mean(-1) < 150
    for x0, y0, x1, y1 in ERASE:
        black[y0:y1, x0:x1] = False
    return black


def to_world(pts, level):
    pts = np.asarray(pts, np.float64)
    dy = PLAT_DY if level == "B2" else 0
    return np.stack([(pts[:, 0] - X0) * S, -(pts[:, 1] - dy - Y0) * S], -1)


def mask_polys(mask, level, eps=0.7, min_area_px=3, up=4):
    big = cv2.resize(mask.astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    cs, hier = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1:
            continue
        if cv2.contourArea(c) < min_area_px * up * up:
            continue
        rings = [c]
        j = hier[0][i][2]
        while j != -1:
            if cv2.contourArea(cs[j]) >= 2 * up * up:
                rings.append(cs[j])
            j = hier[0][j][0]
        conv = []
        for r in rings:
            a = cv2.approxPolyDP(r, eps * up, True)[:, 0, :].astype(np.float64)
            if len(a) < 3:
                continue
            px = (a + 0.5) / up - 0.5
            conv.append(to_world(px, level).round(3).tolist())
        if conv:
            out.append(conv)
    return out


def flood(free, seeds, roi):
    h, w = free.shape
    roi_mask = np.zeros_like(free)
    x0, y0, x1, y1 = roi
    roi_mask[y0:y1, x0:x1] = True
    free = free & roi_mask
    reach = np.zeros((h + 2, w + 2), np.uint8)
    img = np.where(free, 255, 0).astype(np.uint8)
    for sx, sy in seeds:
        if img[sy, sx] == 255 and reach[sy + 1, sx + 1] == 0:
            cv2.floodFill(img, reach, (sx, sy), 128, flags=4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY)
    return reach[1:-1, 1:-1] > 0


def find_hline(black, y_guess, x0, x1, search=4):
    best = None
    for y in range(y_guess - search, y_guess + search + 1):
        run = black[y, x0:x1].mean()
        if best is None or run > best[1]:
            best = (y, run)
    return best[0]


def build():
    black = load()
    walls = black.copy()
    # remove symbols that become separate 3D objects
    for st in STAIRS.values():
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        walls[st["y0"]:st["y1"], xa:xb] = False                                   # B2 foot
        walls[st["y0"] - PLAT_DY:st["y1"] - PLAT_DY, xa:xb] = False               # B1 opening
    for x0, y0, x1, y1 in GATE_ERASE + EXIT_ERASE:
        walls[y0:y1, x0:x1] = False

    # platform edges: long horizontal lines -> platform screen doors instead of walls
    e_up = find_hline(black, 1049, 600, 1500)
    e_lo = find_hline(black, 1203, 600, 1500)
    edges = {"up": e_up, "lo": e_lo}
    plat_walls = walls.copy()
    plat_walls[e_up - 1:e_up + 2, 440:1700] = False
    plat_walls[e_lo - 1:e_lo + 2, 440:1700] = False

    # walkable floors
    free_b1 = ~cv2.dilate(walls.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    vault_b1 = np.zeros_like(free_b1)
    vault_b1[VAULT["y0_plat"] - PLAT_DY:VAULT["y1_plat"] - PLAT_DY, VAULT["x0"]:VAULT["x1"]] = True
    outline = np.zeros(free_b1.shape, np.uint8)
    cv2.fillPoly(outline, [np.array(B1_OUTLINE, np.int32)], 1)
    outline = outline.astype(bool)
    walk_b1 = flood(free_b1 & ~vault_b1 & outline, SEEDS_B1, B1_ROI)
    exit_fp = np.zeros(free_b1.shape, np.uint8)
    for e in EXITS.values():
        c = np.array(e["c"], np.float64); u = np.array(e["up"], np.float64); u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        pts = [c + u * e["run"] / 2 + v * e["w"] / 2, c + u * e["run"] / 2 - v * e["w"] / 2,
               c - u * e["run"] / 2 - v * e["w"] / 2, c - u * e["run"] / 2 + v * e["w"] / 2]
        cv2.fillPoly(exit_fp, [np.array(pts, np.int32)], 1)
    walk_b1 |= exit_fp.astype(bool) & outline
    openings = np.zeros_like(walk_b1)
    for st in STAIRS.values():   # stair openings are holes in the B1 floor
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        openings[st["y0"] - PLAT_DY:st["y1"] - PLAT_DY, xa:xb] = True
    walk_b1 &= ~openings
    free_b2 = ~cv2.dilate(walls.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    walk_b2 = flood(free_b2, SEEDS_B2, (440, e_up + 1, 1700, e_lo))

    # wall solids: drawn lines plus thick-wall infill hugging the walkable area
    def solids(wmask, walk, roi):
        band = cv2.dilate(walk.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        x0, y0, x1, y1 = roi
        keep = np.zeros_like(wmask)
        keep[y0:y1, x0:x1] = True
        sol = (wmask | (band & ~walk)) & keep
        sol &= ~walk
        return cv2.dilate(sol.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & ~walk

    # everything inside the outline that is not floor, void or stair opening is solid (walls, rooms)
    sol_b1 = outline & ~walk_b1 & ~vault_b1 & ~openings
    plat_box = np.zeros_like(walk_b2)
    plat_box[e_up + 2:e_lo - 1, 450:1700] = True
    plat_open = np.zeros_like(walk_b2)
    for st in STAIRS.values():
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        plat_open[st["y0"]:st["y1"], xa:xb] = True
    walk_b2 |= plat_open & plat_box
    sol_b2 = plat_box & ~walk_b2

    # B1 flat ceiling everywhere except under the corridor vault
    ceil_b1 = (walk_b1 | sol_b1).copy()
    cr = B1_CORRIDOR
    ceil_b1[cr["y0"]:cr["y1"], cr["x0"]:cr["x1"]] = False

    # B2 ceiling: over both platform halves and both track beds, minus stair shafts and the vault
    ceil_b2 = np.zeros_like(walk_b2)
    ceil_b2[1028:1222, 440:1700] = True
    ceil_b2 &= ~plat_open
    ceil_b2[VAULT["y0_plat"]:VAULT["y1_plat"], VAULT["x0"]:VAULT["x1"]] = False

    plan = {
        "scale": S, "origin_px": [X0, Y0], "plat_dy": PLAT_DY,
        "b1_floor": mask_polys(walk_b1 | sol_b1, "B1"),
        "b1_walk": mask_polys(walk_b1, "B1"),
        "b1_ceiling": mask_polys(ceil_b1, "B1"),
        "b1_corridor": {"x0": (B1_CORRIDOR["x0"] - X0) * S, "x1": (B1_CORRIDOR["x1"] - X0) * S,
                        "y_n": -(B1_CORRIDOR["y0"] - Y0) * S, "y_s": -(B1_CORRIDOR["y1"] - Y0) * S},
        "b1_walls": mask_polys(sol_b1, "B1"),
        "b2_floor": mask_polys(walk_b2 | sol_b2, "B2"),
        "b2_walk": mask_polys(walk_b2, "B2"),
        "b2_walls": mask_polys(sol_b2, "B2"),
        "b2_ceiling": mask_polys(ceil_b2, "B2"),
        "track_wall_y": {"east": float(to_world([(0, 1028)], "B2")[0, 1]), "west": float(to_world([(0, 1222)], "B2")[0, 1])},
        "edges_y": {k: float(to_world([(0, v)], "B2")[0, 1]) for k, v in edges.items()},
        "plat_x": [float(to_world([(455, 0)], "B2")[0, 0]), float(to_world([(1600, 0)], "B2")[0, 0])],
        "stairs": {}, "gates": {}, "exits": {},
        "vault": {"x0": float((VAULT["x0"] - X0) * S), "x1": float((VAULT["x1"] - X0) * S),
                  "y_east": float(to_world([(0, VAULT["y0_plat"])], "B2")[0, 1]),
                  "y_west": float(to_world([(0, VAULT["y1_plat"])], "B2")[0, 1])},
    }
    for k, st in STAIRS.items():
        a = to_world([(st["x_top"], st["y0"]), (st["x_bot"], st["y1"])], "B2")
        plan["stairs"][k] = {"x_top": a[0, 0], "x_bot": a[1, 0], "y0": min(a[0, 1], a[1, 1]), "y1": max(a[0, 1], a[1, 1])}
    for k, pl in GATES.items():
        plan["gates"][k] = to_world(pl, "B1").round(3).tolist()
    for k, e in EXITS.items():
        c = to_world([e["c"]], "B1")[0]
        u = np.array([e["up"][0], -e["up"][1]], np.float64)
        u /= np.linalg.norm(u)
        plan["exits"][k] = {"c": c.round(3).tolist(), "run": e["run"] * S, "w": e["w"] * S, "up": u.round(4).tolist()}

    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "plan.json"), "w") as f:
        json.dump(plan, f)
    # debug overlay
    dbg = cv2.imread(os.path.join(CACHE, "map.png"))
    ov = dbg.copy()
    ov[walk_b1] = (120, 220, 120)
    ov[walk_b2] = (220, 180, 120)
    ov[sol_b1] = (40, 40, 200)
    ov[sol_b2] = (40, 40, 200)
    ov[vault_b1] = (200, 120, 220)
    dbg = cv2.addWeighted(dbg, 0.35, ov, 0.65, 0)
    for st in STAIRS.values():
        xa, xb = sorted((st["x_top"], st["x_bot"]))
        cv2.rectangle(dbg, (xa, st["y0"]), (xb, st["y1"]), (0, 160, 255), 1)
        cv2.rectangle(dbg, (xa, st["y0"] - PLAT_DY), (xb, st["y1"] - PLAT_DY), (0, 160, 255), 1)
    for pl in GATES.values():
        cv2.polylines(dbg, [np.array(pl, np.int32)], False, (255, 0, 255), 2)
    for e in EXITS.values():
        c = np.array(e["c"], np.float64); u = np.array(e["up"], np.float64); u /= np.linalg.norm(u)
        cv2.arrowedLine(dbg, tuple((c - u * e["run"] / 2).astype(int)), tuple((c + u * e["run"] / 2).astype(int)), (0, 120, 0), 2)
    cv2.polylines(dbg, [np.array(B1_OUTLINE, np.int32)], True, (0, 0, 0), 1)
    cv2.line(dbg, (440, e_up), (1700, e_up), (255, 0, 0), 1)
    cv2.line(dbg, (440, e_lo), (1700, e_lo), (255, 0, 0), 1)
    cv2.imwrite(os.path.join(CACHE, "plan_debug.png"), dbg)
    print("edges px", edges, "b1 walls", len(plan["b1_walls"]), "b2 walls", len(plan["b2_walls"]),
          "walk px", int(walk_b1.sum()), int(walk_b2.sum()))
    return plan


if __name__ == "__main__":
    build()
