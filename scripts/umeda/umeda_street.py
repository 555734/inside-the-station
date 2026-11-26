"""Exits and the street above: each exit is a full stairwell from the concourse (B1) up to street
level (tiled side walls, sloping ceiling with lights, handrails on wall brackets and, on wide
stairs, a centre rail on posts, a landing half-way), ending under an exit canopy on the pavement of
Midosuji. The street: pavements on both sides with ginkgo trees, street lights, guard rails along
the kerb (the walker stays on the pavement), the carriageway with its lane markings, building
frontages on both sides, and beyond the walkable stretch the avenue carries on to the horizon.

build(BU, M) is called from build_umeda after the stations levels exist (it cuts the stairwells
through what is in their way)."""
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

Z_ST = 5.6               # street level above the concourse floor (33 risers of 0.17 m)
H_HOOD = 3.0             # clear height under the exit canopy
STREET_Y = (-15.0, 28.0)  # carriageway between the kerbs (pavement south of the first, north of the second)
FRONT_Y = (-48.0, 46.0)   # building frontages
WALK_X = (-138.0, 138.0)  # pavement stretch the walker can use (site hoardings at both ends)
FAR_X = (-420.0, 420.0)   # the avenue as far as it is seen


def _new_objects(before):
    return [ob for ob in bpy.data.objects if ob.name not in before]


def _frame(origin, u):
    """Local frame of an exit: x up the stair (u), y to its left, z up."""
    ang = math.atan2(u[1], u[0])
    return Matrix.Translation((origin[0], origin[1], 0.0)) @ Matrix.Rotation(ang, 4, "Z")


def materials(BU, B, M):
    T = BU.T
    if "asphalt" not in M:
        M["asphalt"] = _asphalt(B, T)
        M["pavement"] = BU.pavers("pavement", 0.3, 0.3, (0.52, 0.5, 0.47), (0.33, 0.32, 0.3), 0.55, 41)
        M["kerb"] = B.plain("kerb", (0.62, 0.61, 0.58), 0.7)
        M["guard_white"] = B.plain("guard_white", (0.86, 0.86, 0.84), 0.35, coat=0.3)
        M["road_paint"] = B.plain("road_paint", (0.86, 0.86, 0.82), 0.6)
        M["bark"] = B.plain("bark", (0.16, 0.13, 0.1), 0.85)
        M["ginkgo"] = B.plain("ginkgo", (0.3, 0.42, 0.12), 0.75)
        M["pole_grey"] = B.plain("pole_grey", (0.38, 0.39, 0.4), 0.4, metal=0.6)
        M["hood_panel"] = B.plain("hood_panel", (0.72, 0.73, 0.74), 0.35, metal=0.3)
        M["hood_dark"] = B.plain("hood_dark", (0.09, 0.09, 0.1), 0.4)
        M["hoarding"] = B.plain("hoarding", (0.88, 0.9, 0.88), 0.5)
        M["hoarding_green"] = B.plain("hoarding_green", (0.12, 0.42, 0.25), 0.5)
        M["shop_glass"] = B.plain("shop_glass", (0.06, 0.08, 0.09), 0.05, coat=1.0)
        for i, wall in enumerate(((0.62, 0.6, 0.56), (0.42, 0.44, 0.47), (0.7, 0.68, 0.62), (0.3, 0.31, 0.33))):
            M[f"facade{i}"] = _facade(B, T, f"facade{i}", wall, (0.1, 0.13, 0.16), seed=11 + i)
        sky = bpy.data.worlds.get("street_sky") or _sky_world()
        bpy.context.scene.world = sky
    return M


def _asphalt(B, T):
    name = "asphalt"
    path = os.path.join(T.CACHE, "asphalt_1024.npz")
    if not os.path.exists(path):
        N = 1024
        rng = np.random.default_rng(5)
        g = rng.random((N, N)).astype(np.float32)
        from scipy import ndimage
        grit = ndimage.gaussian_filter(g, 0.8)
        cloud = T.fbm((N, N), [20, 80], [0.5, 0.5], 6)
        v = 0.11 + 0.05 * (grit - 0.5) + 0.02 * cloud
        alb = np.stack([v, v * 0.99, v * 0.97], -1)
        rough = np.clip(0.82 + 0.1 * (grit - 0.5), 0.5, 1.0)
        h = (grit - 0.5) * 0.002
        d = {"albedo": alb, "rough": rough.astype(np.float32), "normal": T.height_to_normal(h, 2.4 / N)}
        np.savez(path, **{k: v_.astype(np.float16) for k, v_ in d.items()})
    d = {k: v_.astype(np.float32) for k, v_ in np.load(path).items()}
    return B.tiled(name, B.tex_set(name, d))


def _facade(B, T, name, wall, glass, seed):
    """Office frontage: 1.8 m bays, 3.6 m floors, a window band in every bay."""
    path = os.path.join(T.CACHE, f"{name}_1024.npz")
    if not os.path.exists(path):
        N = 1024
        size = 7.2                                   # texture covers 4 bays x 2 floors
        px = size / N
        c = (np.arange(N, dtype=np.float32) + 0.5) * px
        U, V = np.meshgrid(c, c)
        bu = np.mod(U, 1.8)
        bv = np.mod(V, 3.6)
        win = (bu > 0.15) & (bu < 1.65) & (bv > 0.9) & (bv < 3.3)
        mull = win & (np.abs(bu - 0.9) < 0.03)
        rng = np.random.default_rng(seed)
        tone = 1 + 0.04 * T.fbm((N, N), [30], [1.0], seed)
        alb = np.where(win[..., None], np.array(glass, np.float32) * (1 + 0.3 * rng.random() * 0), np.array(wall, np.float32) * tone[..., None])
        alb = np.where(mull[..., None], np.array((0.25, 0.26, 0.27), np.float32), alb)
        rough = np.where(win, 0.08, 0.55).astype(np.float32)
        h = np.where(win, -0.004, 0.0).astype(np.float32)
        d = {"albedo": alb.astype(np.float32), "rough": rough, "normal": T.height_to_normal(h, px)}
        np.savez(path, **{k: v_.astype(np.float16) for k, v_ in d.items()})
    d = {k: v_.astype(np.float32) for k, v_ in np.load(path).items()}
    return B.tiled(name, B.tex_set(name, d))


def _sky_world():
    w = bpy.data.worlds.new("street_sky")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
    except TypeError:
        pass
    for attr, val in (("sun_elevation", math.radians(38)), ("sun_rotation", math.radians(200)), ("altitude", 40.0)):
        if hasattr(sky, attr):
            setattr(sky, attr, val)
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False        # the sun is a separate lamp (sharp shadows, known strength)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = SKY_STRENGTH
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    sun = bpy.data.lights.new("street_sun", "SUN")
    sun.energy = SUN_STRENGTH
    sun.angle = math.radians(0.6)
    sun.color = (1.0, 0.96, 0.9)
    so = bpy.data.objects.new("street_sun", sun)
    bpy.context.scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(90 - 38), 0, math.radians(200 - 90))
    return w


# outdoor light, kept a few times brighter than the concourse rather than the real 100x, so one
# exposure shows both (the app adapts its exposure a little on top)
SKY_STRENGTH = 0.12
SUN_STRENGTH = 2.8


# ------------------------------------------------------------------ exits

_MASKS = {}


def _b1_masks(BU):
    """Concourse floor and wall masks of the map (0.1 m), for the room beside a stairwell."""
    if not _MASKS:
        import cv2
        res, X0, Y0 = 0.1, -150.0, -70.0
        W, H = int(300 / res), int(140 / res)
        for key in ("b1_floor", "b1_walls"):
            m = np.zeros((H, W), np.uint8)
            for poly in BU.PLAN[key]:
                cv2.fillPoly(m, [np.array([[(x - X0) / res, (y - Y0) / res] for x, y in poly[0]], np.int32)], 1)
            _MASKS[key] = m.astype(bool)
        _MASKS["geo"] = (res, X0, Y0, W, H)
    return _MASKS


def _in_other_exit(BU, p, own_foot):
    """Is plan point p inside the stairwell (with a margin) of an exit other than the one at own_foot?"""
    for e in BU.PLAN["exits"].values():
        u = np.array(e["up"], float)
        u /= np.linalg.norm(u)
        f = np.array(e["c"], float) - u * e["run"] / 2
        if np.linalg.norm(f - own_foot) < 1e-6:
            continue
        w = float(np.clip(e["w"], 2.4, 7.2))
        n = int(round((Z_ST - BU.Z_B1) / 0.17))
        hx = n * 0.30 + 1.6 + 3.2
        q = p - f
        lx, ly = float(q @ u), float(q @ np.array((-u[1], u[0])))
        if -0.8 <= lx <= hx + 0.5 and abs(ly) <= w / 2 + 0.8:
            return True
    return False


def _side_fills(BU, foot, u, ya, yb, xa, xb, narrow=4.5):
    """The map's exit passages are often wider than the stair built in them: beside the stairwell
    a strip of concourse floor would run on as a blind alley under the concourse ceiling. Where
    the floor beside a stairwell wall reaches no further than `narrow` m along the whole flight,
    that strip is solid (a block out to the floor edge); wider floor there is open concourse and
    stays. Returns (y_lo, y_hi) in the stair frame per side, or None."""
    m = _b1_masks(BU)
    res, X0, Y0, W, H = m["geo"]
    v = np.array((-u[1], u[0]))
    out = []
    for side, y_edge in ((-1, ya), (1, yb)):
        ext = []
        for x in np.arange(xa + 0.6, xb - 0.3, 0.5):
            d = 0.0
            while d < narrow + 1.0:
                p = foot + u * x + v * (y_edge + side * (d + 0.05))
                ix, iy = int((p[0] - X0) / res), int((p[1] - Y0) / res)
                if not (0 <= ix < W and 0 <= iy < H) or not m["b1_floor"][iy, ix] or m["b1_walls"][iy, ix]:
                    break
                if _in_other_exit(BU, p, foot):
                    break
                d += 0.1
            ext.append(d)
        e = max(ext) if ext else 0.0
        if 0.15 < e <= narrow:
            lo, hi = sorted((y_edge, y_edge + side * (e + 0.3)))
            out.append((lo, hi))
        else:
            out.append((None, None))
    return out


def exit_stair(BU, B, M, k, e):
    """Stairwell from the concourse up to the street, in the exit's own frame."""
    c = np.array(e["c"], float)
    u = np.array(e["up"], float)
    u /= np.linalg.norm(u)
    w = float(np.clip(e["w"], 2.4, 7.2))
    foot = c - u * e["run"] / 2
    n = int(round((Z_ST - BU.Z_B1) / 0.17))
    riser = (Z_ST - BU.Z_B1) / n
    tread = 0.30
    land = 1.6
    L = n * tread + land                   # run of the whole stair
    y0, y1 = -w / 2, w / 2
    hood_x1 = L + 3.2                      # canopy floor beyond the last step
    before = set(o.name for o in bpy.data.objects)
    nramps = len(BU.RAMPS)
    cname = "Street"
    BU.flight(f"ex{k}", L, 0.0, y0, y1, Z_ST, BU.Z_B1, M, cname, landing=True)
    if w > 4.0:
        pass                               # flight() already sets a centre rail on posts

    # nosing line of the flight (with the landing) and the ceiling over it
    path = BU.RAMPS[nramps][2]

    def nose(x):
        for (xa, za), (xb, zb) in zip(path[:-1], path[1:]):
            lo, hi = sorted((xa, xb))
            if lo - 1e-6 <= x <= hi + 1e-6:
                return za if abs(xb - xa) < 1e-9 else za + (x - xa) / (xb - xa) * (zb - za)
        return BU.Z_B1 if x < 0 else Z_ST

    xs = np.linspace(-0.3, L + 0.3, 120)
    ceil = np.array([max(BU.Z_B1 + BU.H_B1, nose(x) + 2.7) for x in xs])
    # the stairwell is roofed by the canopy where its ceiling would come out of the ground
    x_open = float(xs[np.argmax(ceil >= Z_ST - 0.05)])
    # side walls: from the concourse floor (the space under the flight is closed, not open to
    # whatever floor the map has beside the stairwell) up to the ceiling line, or the street
    # where the canopy starts; an end wall under the canopy floor
    for yy, nm in ((y0 - 0.15, "a"), (y1, "b")):
        top = [(x, min(cz, Z_ST)) for x, cz in zip(xs, ceil)] + [(hood_x1, Z_ST)]
        bot = [(hood_x1, BU.Z_B1), (float(xs[0]), BU.Z_B1)]
        BU.incline_prism(f"ex{k}_wall{nm}", top + bot, yy, yy + 0.15, M["wall_tile"], cname)
    BU.bx(f"ex{k}_under_end", (hood_x1, y0 - 0.15, BU.Z_B1), (hood_x1 + 0.15, y1 + 0.15, Z_ST - 0.35), M["wall_tile"], cname)
    fills = _side_fills(BU, foot, u, y0 - 0.15, y1 + 0.15, float(xs[0]), hood_x1 + 0.15)
    for (ya_, yb_), nm in zip(fills, ("a", "b")):
        if ya_ is not None:
            BU.bx(f"ex{k}_fill{nm}", (float(xs[0]), ya_, BU.Z_B1), (hood_x1 + 0.15, yb_, BU.Z_B1 + BU.H_B1 + 0.04), M["b1_wall"], cname)
            BU.EXIT_FILLS.append((_frame(foot, u), (float(xs[0]), ya_, BU.Z_B1), (hood_x1 + 0.15, yb_, BU.Z_B1 + BU.H_B1 + 0.04)))
    # sloping ceiling over the stair (up to the canopy) and its lights
    xc = [x for x in xs if x <= x_open]
    pts = [(x, max(BU.Z_B1 + BU.H_B1, nose(x) + 2.7)) for x in xc]
    ceil_poly = pts + [(x, z + 0.08) for (x, z) in pts[::-1]]
    BU.incline_prism(f"ex{k}_ceil", ceil_poly, y0, y1, M["ceil"], cname)
    xl = 1.2
    j = 0
    while xl < x_open - 0.8:
        zc = max(BU.Z_B1 + BU.H_B1, nose(xl) + 2.7)
        BU.bx(f"ex{k}_led{j}", (xl - 0.6, -0.07, zc - 0.04), (xl + 0.6, 0.07, zc), M["led_line"], "Street_lights")
        L_ = B.area_light(f"ex{k}_ledA{j}", (xl, 0.0, zc - 0.05), 1.2, 0.14, 70.0, (1.0, 0.97, 0.92), camera=False)
        BU.put(L_, "Street_lights")
        xl += 2.4
        j += 1
    # canopy over the top of the stair: walls, roof, fascia, lights
    yh0, yh1 = y0 - 0.15, y1 + 0.15
    BU.bx(f"ex{k}_hood_back", (x_open - 0.3, yh0, Z_ST - 0.02), (x_open, yh1, Z_ST + H_HOOD), M["hood_panel"], cname)
    for yy, nm in ((yh0 - 0.12, "a"), (yh1, "b")):
        BU.bx(f"ex{k}_hood_side{nm}", (x_open - 0.3, yy, Z_ST - 0.35), (hood_x1, yy + 0.12, Z_ST + H_HOOD), M["hood_panel"], cname)
        BU.bx(f"ex{k}_hood_plinth{nm}", (x_open - 0.3, yy - 0.01, Z_ST), (hood_x1, yy + 0.13, Z_ST + 0.25), M["hood_dark"], cname)
    BU.bx(f"ex{k}_hood_roof", (x_open - 0.5, yh0 - 0.4, Z_ST + H_HOOD), (hood_x1 + 0.5, yh1 + 0.4, Z_ST + H_HOOD + 0.35), M["hood_panel"], cname)
    BU.bx(f"ex{k}_hood_fascia", (hood_x1 + 0.5, yh0 - 0.4, Z_ST + H_HOOD - 0.05), (hood_x1 + 0.56, yh1 + 0.4, Z_ST + H_HOOD + 0.4), M["hood_dark"], cname)
    BU.bx(f"ex{k}_hood_floor", (L, y0, Z_ST - 0.35), (hood_x1, y1, Z_ST), M["pavement"], cname)
    xl = x_open + 1.2
    j = 0
    while xl < hood_x1 - 0.5:
        for yl in (y0 + 0.8, y1 - 0.8) if w > 3.0 else (0.0,):
            BU.put(B.cylinder(f"ex{k}_dl{j}", (xl, yl, Z_ST + H_HOOD - 0.004), 0.09, 0.01, M["downlight"], segs=20), "Street_lights")
            L_ = B.area_light(f"ex{k}_dlA{j}", (xl, yl, Z_ST + H_HOOD - 0.02), 0.2, 0.2, 85.0, (1.0, 0.96, 0.9), camera=False)
            L_.data.spread = math.radians(150)
            BU.put(L_, "Street_lights")
            j += 1
        xl += 2.4
    # into world space
    F = _frame(foot, u)
    for ob in _new_objects(before):
        if ob.type in ("MESH", "LIGHT"):
            ob.matrix_world = F @ ob.matrix_world
    y_a, y_b, rp = BU.RAMPS[nramps]
    BU.RAMPS[nramps] = (y_a, y_b, rp, (float(foot[0]), float(foot[1]), math.atan2(u[1], u[0])))
    # what the stairwell and canopy must cut away: everything of the station above the concourse
    # ceiling inside the stairwell, and the pavement over the opening
    # the whole stairwell, from just above the concourse floor: walls the map drew over the exit
    # (some exits were vectorised as a solid block) and the concourse ceiling over the stair go
    cut = (F, (-0.25, y0 - 0.14, BU.Z_B1 + 0.01), (hood_x1, y1 + 0.14, Z_ST + H_HOOD))
    hood = (F, x_open - 0.3, hood_x1 + 0.6, yh0 - 0.45, yh1 + 0.45)
    return cut, hood


def carve(BU, cuts, keep_prefix=("ex",)):
    """Boolean the stairwell volumes out of the station objects they run through."""
    for F, lo, hi in cuts:
        me = bpy.data.meshes.new("cutter")
        bx_ = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
               (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
        me.from_pydata(bx_, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        cutter = bpy.data.objects.new("cutter", me)
        bpy.context.scene.collection.objects.link(cutter)
        cutter.matrix_world = F
        corners = [F @ Vector(v) for v in bx_]
        cmin = Vector([min(p[i] for p in corners) for i in range(3)])
        cmax = Vector([max(p[i] for p in corners) for i in range(3)])
        for ob in list(bpy.data.objects):
            if ob.type != "MESH" or ob is cutter or ob.name.startswith(keep_prefix) or not ob.data.polygons:
                continue
            if any(c.name.startswith("Street") for c in ob.users_collection):
                continue
            bb = [ob.matrix_world @ Vector(v) for v in ob.bound_box]
            bmin = Vector([min(p[i] for p in bb) for i in range(3)])
            bmax = Vector([max(p[i] for p in bb) for i in range(3)])
            if any(bmax[i] < cmin[i] or bmin[i] > cmax[i] for i in range(3)):
                continue
            mod = ob.modifiers.new("cut", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.solver = "EXACT"
            mod.object = cutter
            bpy.context.view_layer.objects.active = ob
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError:
                ob.modifiers.remove(mod)
        bpy.data.objects.remove(cutter)


# ------------------------------------------------------------------ street

def street(BU, B, M, hoods):
    """Near part (the walkable stretch and a little beyond, baked) in 'Street'; the avenue further
    on, seen but never reached, in 'Street_far' (no lightmap, lit by the sky in the app)."""
    z = Z_ST
    ys, yn = STREET_Y
    fs, fn = FRONT_Y
    wx0, wx1 = WALK_X
    nx0, nx1 = wx0 - 12.0, wx1 + 12.0         # baked stretch
    fx0, fx1 = FAR_X
    for xa, xb, c in ((nx0, nx1, "Street"), (fx0, nx0, "Street_far"), (nx1, fx1, "Street_far")):
        tag = "" if c == "Street" else ("_w" if xa < 0 else "_e")
        for (ya, yb, nm) in ((fs, ys, "s"), (yn, fn, "n")):
            if c == "Street":
                _pavement(BU, B, M, f"pave_{nm}", xa, xb, ya, yb, z, hoods, c)
            else:
                BU.bx(f"pave_{nm}{tag}", (xa, ya, z - 0.5), (xb, yb, z), M["pavement"], c)
        BU.bx(f"carriageway{tag}", (xa, ys, z - 0.5), (xb, yn, z - 0.15), M["asphalt"], c)
        for yy, nm in ((ys, "s"), (yn, "n")):
            a, b = sorted((yy, yy + (0.2 if nm == "s" else -0.2)))
            BU.bx(f"kerb_{nm}{tag}", (xa, a, z - 0.5), (xb, b, z + 0.005), M["kerb"], c)
        # lane lines: Midosuji runs one way, six lanes
        lanes = 6
        lw = (yn - ys - 2.0) / lanes
        for i in range(1, lanes):
            yl = ys + 1.0 + i * lw
            x = xa + 2.0
            while x + 6.0 < xb:
                BU.bx(f"lane_{i}_{x:.0f}", (x, yl - 0.075, z - 0.15), (x + 6.0, yl + 0.075, z - 0.145), M["road_paint"], c)
                x += 15.0
        for yy in (ys + 0.6, yn - 0.6):
            BU.bx(f"edge_line_{yy:.0f}{tag}", (xa, yy - 0.1, z - 0.15), (xb, yy + 0.1, z - 0.145), M["road_paint"], c)
        # trees and lamps on both pavements
        for yy, side in ((ys - 0.45, -1), (yn + 0.45, 1)):
            x = xa + 5.0
            k = 0
            while x < xb:
                yt = yy + side * 1.4
                if not _near_hood(x, yt, hoods, 3.0):
                    _ginkgo(BU, B, M, f"tree_{side}_{x:.0f}", x, yt, z, c, seed=abs(int(x * 7)) * 2 + (side > 0))
                if k % 2 == 0 and not _near_hood(x + 5, yt, hoods, 2.0):
                    _street_lamp(BU, B, M, f"lamp_{side}_{x:.0f}", x + 5, yt, z, -side, c)
                k += 1
                x += 10.0
        # frontages: one face per building (what the street sees), shop fronts below
        rng = np.random.default_rng(3 if c == "Street" else (5 if xa < 0 else 7))
        for yf, side in ((fs, -1), (fn, 1)):
            x = xa
            k = 0
            while x < xb - 1.0:
                wb = min(float(rng.uniform(18, 42)), xb - x)
                hb = float(rng.uniform(24, 70))
                mat = M[f"facade{(k + (side > 0)) % 4}"]
                nrm = (0, -side, 0)
                BU.vrect(f"bldg_{side}{tag}_{k}", (x, yf), (x + wb, yf), z, z + hb, mat, c, nrm, uv_scale=1.0)
                ga, gb = sorted((yf - side * 0.02, yf - side * 0.06))
                BU.bx(f"shop_{side}{tag}_{k}", (x + 1.0, ga, z), (x + wb - 1.3, gb, z + 3.6), M["shop_glass"], c)
                sa, sb = sorted((yf, yf - side * 0.35))
                BU.bx(f"shopband_{side}{tag}_{k}", (x + 0.5, sa, z + 3.6), (x + wb - 0.8, sb, z + 4.2), M["hood_dark"], c)
                x += wb
                k += 1
    # guard rails along both kerbs over the walkable stretch (white pipe rails)
    c = "Street"
    _furniture(BU, B, M, hoods, c)
    for yy, side in ((ys - 0.45, -1), (yn + 0.45, 1)):
        x = wx0
        k = 0
        while x <= wx1:
            BU.put(B.cylinder(f"gr_post_{side}_{k}", (x, yy, z + 0.4), 0.035, 0.8, M["guard_white"], segs=10), c)
            k += 1
            x += 2.0
        for h in (0.35, 0.6, 0.8):
            BU.slope_rail(f"gr_pipe_{side}_{h}", (wx0, yy, z + h), (wx1, yy, z + h), 0.024, M["guard_white"], c)
    # site hoardings closing the pavements at both ends of the walkable stretch
    for xh, nm in ((wx0, "w"), (wx1, "e")):
        for (ya, yb, sn) in ((fs, ys - 0.2, "s"), (yn + 0.2, fn, "n")):
            BU.bx(f"hoard_{nm}{sn}", (xh - 0.06, ya, z), (xh + 0.06, yb, z + 2.4), M["hoarding"], c)
            BU.bx(f"hoardb_{nm}{sn}", (xh - 0.07, ya, z + 2.2), (xh + 0.07, yb, z + 2.4), M["hoarding_green"], c)
    # the far ends of the avenue: it bends out of sight behind a block
    for xe, sgn in ((fx0, 1), (fx1, -1)):
        BU.vrect(f"bldg_end{sgn}", (xe, fs), (xe, fn), z, z + 60, M["facade1"], "Street_far", (sgn, 0, 0))


def _near_hood(x, y, hoods, r):
    for F, xa, xb, ya, yb in hoods:
        p = F.inverted() @ Vector((x, y, 0.0))
        if xa - r < p.x < xb + r and ya - r < p.y < yb + r:
            return True
    return False


def _pavement(BU, B, M, name, x0, x1, ya, yb, z, hoods, c):
    """Pavement slab cut around the canopies (whose floor carries on into them)."""
    ob = BU.bx(name, (x0, ya, z - 0.5), (x1, yb, z), M["pavement"], c)
    for i, (F, xa, xb, ha, hb) in enumerate(hoods):
        me = bpy.data.meshes.new("hcut")
        lo, hi = (xa, ha + 0.3, z - 1.0), (xb - 0.6, hb - 0.3, z + 1.0)
        v = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
             (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
        me.from_pydata(v, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        cut = bpy.data.objects.new("hcut", me)
        bpy.context.scene.collection.objects.link(cut)
        cut.matrix_world = Matrix.Translation((0, 0, 0)) @ F
        mod = ob.modifiers.new("cut", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = cut
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cut)
    return ob


def _ginkgo(BU, B, M, name, x, y, z, c, seed):
    """Street ginkgo: straight trunk, an upright narrow crown (a tall ellipsoid with a few lobes)."""
    rng = np.random.default_rng(seed)
    h = float(rng.uniform(7.5, 10.0))
    BU.put(B.cylinder(name + "_trunk", (x, y, z + h * 0.32), 0.16, h * 0.64, M["bark"], segs=10), c)
    BU.bx(name + "_pit", (x - 0.7, y - 0.7, z - 0.02), (x + 0.7, y + 0.7, z + 0.01), M["kerb"], c)
    r = float(rng.uniform(1.6, 2.0))
    BU.put(B.sphere(name + "_crown", (x, y, z + h * 0.6), r, M["ginkgo"], scale=(1.0, 1.0, h * 0.42 / r), segs=14), c)
    for i in range(3):
        a = rng.uniform(0, 2 * math.pi)
        rr = r * float(rng.uniform(0.45, 0.6))
        zz = z + h * float(rng.uniform(0.45, 0.8))
        BU.put(B.sphere(f"{name}_lobe{i}", (x + math.cos(a) * r * 0.6, y + math.sin(a) * r * 0.6, zz), rr, M["ginkgo"],
                        scale=(1, 1, 1.3), segs=10), c)


def _furniture(BU, B, M, hoods, c):
    """Planters with clipped shrubs and benches on the wide south pavement, over the walkable stretch."""
    ys = STREET_Y[0]
    fs = FRONT_Y[0]
    rows = [ys - 9.0, (fs + ys) / 2 - 4.0]
    k = 0
    for yr in rows:
        x = WALK_X[0] + 12.0
        while x < WALK_X[1] - 12.0:
            if not _near_hood(x, yr, hoods, 4.0):
                BU.bx(f"planter_{k}", (x - 2.0, yr - 0.6, Z_ST), (x + 2.0, yr + 0.6, Z_ST + 0.5), M["kerb"], c)
                BU.bx(f"shrub_{k}", (x - 1.9, yr - 0.5, Z_ST + 0.5), (x + 1.9, yr + 0.5, Z_ST + 0.95), M["ginkgo"], c, 0.15)
                BU.bx(f"bench_{k}", (x - 0.9, yr + 1.2, Z_ST + 0.38), (x + 0.9, yr + 1.6, Z_ST + 0.45), M["bark"], c, 0.01)
                for bxp in (-0.75, 0.75):
                    BU.bx(f"benchleg_{k}_{bxp}", (x + bxp - 0.03, yr + 1.25, Z_ST), (x + bxp + 0.03, yr + 1.55, Z_ST + 0.38), M["pole_grey"], c)
                k += 1
            x += 24.0


def _street_lamp(BU, B, M, name, x, y, z, toward, c):
    BU.put(B.cylinder(name + "_pole", (x, y, z + 4.5), 0.08, 9.0, M["pole_grey"], segs=12), c)
    BU.slope_rail(name + "_arm", (x, y, z + 8.8), (x, y + toward * 2.2, z + 9.1), 0.05, M["pole_grey"], c)
    BU.bx(name + "_head", (x - 0.25, y + toward * 2.0 - 0.18, z + 8.95), (x + 0.25, y + toward * 2.6 + 0.18, z + 9.15), M["pole_grey"], c)


def build(BU, M):
    import build_scene as B
    materials(BU, B, M)
    cuts, hoods = [], []
    for k, e in BU.PLAN["exits"].items():
        cut, hood = exit_stair(BU, B, M, k, e)
        cuts.append(cut)
        hoods.append(hood)
    BU.EXIT_CUTS.extend(cuts)
    import time
    t0 = time.time()
    carve(BU, cuts)
    print(f"exits {len(cuts)} carved {time.time() - t0:.0f}s", flush=True)
    street(BU, B, M, hoods)
    print(f"street {time.time() - t0:.0f}s", flush=True)
