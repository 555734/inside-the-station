"""Small fixtures that give the Umeda scenes their density of detail at close range:
fire-hydrant cabinets, AED boxes, emergency-stop buttons on columns, extinguishers,
inspection plates and expansion joints in the floor, smoke detectors on the ceilings.
Positions follow the plan (walls, columns, corridor faces); none of them is on the map itself.
"""
import math

import numpy as np

H = None


def _q(name, c, right, up, w, h, mat, cname):
    H.put(H.B.image_quad(name, c, right, up, w, h, mat), cname)


def hydrant(name, x, y, face, M, cname, zf):
    """Stainless cabinet with a red sign and a red lamp; 'face' = +1/-1 along y."""
    d = 0.12
    y0, y1 = sorted((y, y + face * d))
    H.bx(name, (x - 0.45, y0, zf + 0.25), (x + 0.45, y1, zf + 1.65), M["steel_brushed"], cname, 0.008)
    yf = y + face * (d + 0.002)
    right = (-face, 0, 0)
    _q(name + "_sign", (x, yf, zf + 1.45), right, (0, 0, 1), 0.62, 0.2, M["hydrant"], cname)
    H.bx(name + "_hdl", (x + 0.30, min(yf, yf + face * 0.03), zf + 0.85), (x + 0.33, max(yf, yf + face * 0.03), zf + 1.05), M["steel"], cname)
    H.put(H.B.sphere(name + "_lamp", (x, y + face * 0.06, zf + 1.65 + 0.055), 0.06, M["lamp_red"], scale=(1, 0.6, 1)), cname)


def aed(name, x, y, face, M, cname, zf):
    d = 0.18
    y0, y1 = sorted((y, y + face * d))
    H.bx(name, (x - 0.28, y0, zf + 0.9), (x + 0.28, y1, zf + 1.55), M["kiosk_white"], cname, 0.01)
    _q(name + "_sign", (x, y + face * (d + 0.002), zf + 1.30), (-face, 0, 0), (0, 0, 1), 0.48, 0.36, M["aed"], cname)


def extinguisher(name, x, y, face, M, cname, zf):
    """Red extinguisher standing on the floor against the wall, its 消火器 sign on the wall above."""
    cy = y + face * 0.088
    H.put(H.B.cylinder(name, (x, cy, zf + 0.275), 0.085, 0.55, M["paint_red"], segs=24), cname)
    H.put(H.B.cylinder(name + "_top", (x, cy, zf + 0.59), 0.03, 0.09, M["handrail"], segs=12), cname)
    _q(name + "_sign", (x, y + face * 0.003, zf + 1.05), (-face, 0, 0), (0, 0, 1), 0.36, 0.12, M["fire_ext"], cname)


def estop_on_column(name, cx, cy, r, toward, M, cname, zf):
    """Emergency-stop button box strapped to a round column, facing the platform edge."""
    ang = math.atan2(toward[1], toward[0])
    px, py = cx + math.cos(ang) * (r + 0.04), cy + math.sin(ang) * (r + 0.04)
    ob = H.B.box(name, (0, 0, 0), (0.08, 0.26, 0.36), M["paint_yellow"], 0.01)
    ob.location = (px, py, zf + 1.35)
    ob.rotation_euler = (0, 0, ang)
    H.put(ob, cname)
    fx, fy = cx + math.cos(ang) * (r + 0.081), cy + math.sin(ang) * (r + 0.081)
    right = (-math.sin(ang), math.cos(ang), 0)
    _q(name + "_f", (fx, fy, zf + 1.35), right, (0, 0, 1), 0.24, 0.32, M["estop"], cname)


def floor_plate(name, x, y, M, cname, zf, s=0.6):
    H.bx(name + "_frame", (x - s / 2 - 0.02, y - s / 2 - 0.02, zf), (x + s / 2 + 0.02, y + s / 2 + 0.02, zf + 0.0035), M["steel_brushed"], cname)
    H.bx(name, (x - s / 2, y - s / 2, zf), (x + s / 2, y + s / 2, zf + 0.004), M["checker"], cname)


def smoke_detector(name, x, y, z, M, cname):
    H.put(H.B.cylinder(name, (x, y, z - 0.02), 0.055, 0.04, M["kiosk_white"], segs=24), cname)
    H.put(H.B.cylinder(name + "_l", (x + 0.03, y, z - 0.041), 0.004, 0.003, M["lamp_red"], segs=8), cname)


def build(h, M):
    global H
    H = h
    import cv2
    rng = np.random.default_rng(31)
    P = H.PLAN
    # --- B1 north corridor walls: hydrants, AED, extinguishers between the ad frames
    cr = P["b1_corridor"]
    k = 0
    for i, x in enumerate(np.arange(cr["x0"] + 7.75, cr["x1"] - 3, 11.0)):
        for (y, face, nm) in ((cr["y_n"] - 0.45, -1, "n"), (cr["y_s"] + 0.25, 1, "s")):
            kind = (i + (nm == "s")) % 3
            if kind == 0:
                hydrant(f"hyd1_{k}", x, y, face, M, "B1", H.Z_B1)
            elif kind == 1:
                aed(f"aed1_{k}", x, y, face, M, "B1", H.Z_B1)
            else:
                extinguisher(f"ext1_{k}", x, y, face, M, "B1", H.Z_B1)
            k += 1
    # --- under the arch: hydrant cabinets on the central wall between arches
    V = H.VAULT
    if V is not None:
        # between each arched opening and the poster before it (openings at x0+5+9k, 1.6 m wide)
        for j, x in enumerate(np.arange(V.x0 + 5.0, V.x1 - 3, 9.0)):
            hydrant(f"hydv_{j}", x - 1.5, V.ye, -1, M, "B2", H.Z_B2)
    # --- emergency-stop buttons on round columns near the track-1 edge
    walls = P["b2_walls"]
    ye = P["edges_y"]["up"]
    for i, poly in enumerate(walls):
        r = np.array(poly[0])
        lo, hi = r.min(0), r.max(0)
        d = hi - lo
        c = (lo + hi) / 2
        if max(d) <= 1.0 and min(d) >= 0.6 and c[1] > 13.0 and i % 2 == 0:
            estop_on_column(f"estop_{i}", c[0], c[1], (min(d) - 0.22) / 2, (0.0, 1.0), M, "B2", H.Z_B2)
    # --- inspection plates and expansion joints in the platform floor
    walk2 = [np.array(p[0], np.float32).reshape(-1, 1, 2) for p in P["b2_walk"]]
    k = 0
    for x in np.arange(-95, 108, 17.0):
        for y in (12.0, -1.0, -6.0):
            xx = x + rng.uniform(-3, 3)
            if any(cv2.pointPolygonTest(w, (float(xx), float(y)), False) >= 0 for w in walk2):
                floor_plate(f"plate_{k}", xx, y, M, "B2", H.Z_B2)
                k += 1
    px0, px1 = P["plat_x"]
    for x in np.arange(px0 + 12, px1 - 5, 24.0):
        H.bx(f"joint_{x:.0f}", (x - 0.015, P["edges_y"]["lo"] + 0.4, H.Z_B2), (x + 0.015, ye - 0.4, H.Z_B2 + 0.002), M["steel_brushed"], "B2")
    # --- smoke detectors on the flat ceilings
    k = 0
    for x in np.arange(-100, 110, 7.5):
        for y in (14.5, 10.5, 2.5, -4.0, -9.0):
            if V is not None and V.x0 - 1 < x < V.x1 + 1 and y < 1.5:
                continue
            xx = x + rng.uniform(-1, 1)
            if any(cv2.pointPolygonTest(w, (float(xx), float(y)), False) >= 0 for w in walk2):
                smoke_detector(f"smk_{k}", xx, y, H.Z_B2_CEIL, M, "B2")
                k += 1
    walk1 = [np.array(p[0], np.float32).reshape(-1, 1, 2) for p in P["b1_walk"]]
    for x in np.arange(-140, 130, 8.0):
        for y in (-12.0, -1.0, 8.0):
            xx = x + rng.uniform(-1.5, 1.5)
            if cr["x0"] < xx < cr["x1"] and cr["y_s"] < y < cr["y_n"]:
                continue
            if any(cv2.pointPolygonTest(w, (float(xx), float(y)), False) >= 0 for w in walk1):
                smoke_detector(f"smk1_{k}", xx, y, H.Z_B1 + H.H_B1, M, "B1")
                k += 1
    print("props", k)
