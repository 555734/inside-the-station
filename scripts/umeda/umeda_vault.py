"""The arched (vaulted) part of the track-2 side of Umeda Station's platform, as rebuilt in 2015.

Reference photos (Wikimedia Commons 2019-2024): an elliptical barrel vault over the
track-2 half of the platform. Along its crown hangs a row of mirror-finish stainless light
boxes with a 3x3 grid of etched glass underneath. Each box sits at the bottom of a
funnel-shaped shell clad in white pyramid-relief panels. The pyramid rows radiate from the
box, and neighbouring shells meet along curved folds. The rest of the vault is smooth white
panels with small round downlights.
On the central-wall side the 40 m x 4 m LED screen (Umeda Metro Vision, 2019) follows the
curve. On the track side, a band of backlit ads runs above the tiled track wall.

Everything here is generated from formulas; the shell layout follows the six signage
pillars that the official map shows in the vaulted area.
"""
import math
import os

import bpy
import numpy as np

H = None          # build_umeda module, injected by build()
PX = 0.0125       # vault texture resolution (m per texel)


def setup(h):
    global H
    H = h


# ------------------------------------------------------------------ geometry definition

class Vault:
    def __init__(self, plan, Z_B2):
        v = plan["vault"]
        self.x0, self.x1 = v["x0"], v["x1"]
        self.yw = plan["track_wall_y"]["west"]
        self.ye = v["y_east"] - 0.22                 # face of the central wall
        self.zs = Z_B2 + 3.2                         # springing (top of the track wall)
        self.zc = Z_B2 + 8.05                        # crown of the barrel
        self.a = (self.ye - self.yw) / 2
        self.b = self.zc - self.zs
        self.ym = (self.yw + self.ye) / 2
        t = np.linspace(0.0, math.pi, 6001)
        self.T = t
        self.Y = self.ym - self.a * np.cos(t)
        self.Z = self.zs + self.b * np.sin(t)
        ds = np.hypot(np.diff(self.Y), np.diff(self.Z))
        self.S = np.concatenate([[0.0], np.cumsum(ds)])
        self.Stot = float(self.S[-1])
        # chandeliers / shells: centred over the signage pillars drawn on the map
        cols = []
        for poly in plan["b2_walls"]:
            r = np.array(poly[0])
            c = r.mean(0)
            if self.x0 < c[0] < self.x1 and self.yw < c[1] < self.ye:
                cols.append(((r.min(0) + r.max(0)) / 2))
        cols.sort(key=lambda c: c[0])
        self.kiosks = [(float(c[0]), float(c[1])) for c in cols]
        xs = [c[0] for c in cols]
        pitch = (xs[-1] - xs[0]) / (len(xs) - 1)
        while xs[-1] + pitch < self.x1 - 3.5:       # the row of boxes continues over stair J
            xs.append(xs[-1] + pitch)
        self.cx = np.array(xs)
        self.cy = float(np.mean([c[1] for c in cols]))
        self.box_half = 1.8                          # 3.6 m square light boxes
        self.box_h = 0.6
        self.z_top = Z_B2 + 5.5                      # top of the boxes = bottom of the funnels
        self.fun_A = 2.55                            # funnel rise above the box
        self.fun_R = 3.4                             # horizontal reach of the flare
        self.fun_p = 3.0                             # squarish near the box, rounder further out
        # the LED screen and the ad band, as arc-length ranges
        self.led_s = (self.Stot - 0.30 - 4.0, self.Stot - 0.30)
        self.ad_s = (0.30, 1.90)

    # barrel as a function of arc length
    def at_s(self, s):
        return np.interp(s, self.S, self.Y), np.interp(s, self.S, self.Z)

    def tangent_s(self, s):
        t = np.interp(s, self.S, self.T)
        ty, tz = self.a * np.sin(t), self.b * np.cos(t)
        n = np.hypot(ty, tz)
        return ty / n, tz / n

    def s_of_y(self, y):
        return np.interp(y, self.Y, self.S)

    def funnel(self, x, y):
        """Lowest funnel height at (x, y), index of that funnel, and its distance field."""
        zbest = np.full(np.shape(x), np.inf, np.float64)
        kbest = np.zeros(np.shape(x), np.int32)
        dbest = np.zeros(np.shape(x), np.float64)
        p = self.fun_p
        for k, cx in enumerate(self.cx):
            dx, dy = np.abs(x - cx), np.abs(y - self.cy)
            d = (dx ** p + dy ** p) ** (1.0 / p)
            q = np.clip((d - self.box_half) / self.fun_R, 0.0, 1.0)
            z = self.z_top + self.fun_A * (1.0 - (1.0 - q) ** 2)
            m = z < zbest
            zbest = np.where(m, z, zbest)
            kbest = np.where(m, k, kbest)
            dbest = np.where(m, d, dbest)
        return zbest, kbest, dbest

    def surface(self, x, s):
        y, zb = self.at_s(s)
        zf, k, d = self.funnel(x, y)
        return y, np.minimum(zb, zf), zf < zb, k, d


# ------------------------------------------------------------------ texture (albedo, roughness, object-space normal)

def bake(V, cache):
    path = os.path.join(cache, f"vault_tex_{int(PX * 1e4)}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k].astype(np.float32) for k in d}
    nx = int(math.ceil((V.x1 - V.x0) / PX))
    ns = int(math.ceil(V.Stot / PX))
    xs = V.x0 + (np.arange(nx) + 0.5) * PX
    ss = (np.arange(ns) + 0.5) * PX
    X, Sg = np.meshgrid(xs, ss)                     # rows = s (v), cols = x (u)
    Y, Z, isf, K, D = V.surface(X, Sg)
    # base surface normal (pointing into the room)
    ty, tz = V.tangent_s(Sg)
    nb = np.stack([np.zeros_like(Y), tz, -ty], -1)  # barrel
    cx = V.cx[K]
    dx, dy = X - cx, Y - V.cy
    p = V.fun_p
    adx, ady = np.abs(dx) + 1e-9, np.abs(dy) + 1e-9
    dd = (adx ** p + ady ** p) ** (1.0 / p)
    q = np.clip((dd - V.box_half) / V.fun_R, 0.0, 1.0)
    dz_dd = 2.0 * V.fun_A / V.fun_R * (1.0 - q)
    gx = dz_dd * (adx / dd) ** (p - 1) * np.sign(dx)
    gy = dz_dd * (ady / dd) ** (p - 1) * np.sign(dy)
    nf = np.stack([gx, gy, -np.ones_like(gx)], -1)
    nf /= np.linalg.norm(nf, axis=-1, keepdims=True)
    n0 = np.where(isf[..., None], nf, nb)
    # pyramid relief on the funnels, laid out in log-polar rows around each box
    # diamonds: the lattice is turned 45 degrees, so the ridges run along the rays and rings
    N = 84
    dl = 2 * math.pi / N
    rho = np.log(np.maximum(dd, V.box_half)) - math.log(V.box_half)
    th = np.arctan2(dy, dx) + math.pi
    pu, pv = rho / dl + th / dl, rho / dl - th / dl
    fu = np.mod(pu, 1.0)
    fv = np.mod(pv, 1.0)
    cell = np.maximum(dd, V.box_half) * dl / math.sqrt(2.0)
    tri = np.maximum(np.abs(fu - 0.5), np.abs(fv - 0.5))
    hp = (1.0 - 2.0 * tri) * cell * 0.45
    h = np.where(isf, hp, 0.0)
    # flat panels with joints on the barrel (1.2 m along the vault, 0.9 m around it)
    ju = np.abs(np.mod(X - V.x0 + 0.6, 1.2) - 0.6)
    jv = np.abs(np.mod(Sg + 0.45, 0.9) - 0.45)
    joint = (~isf) & (np.minimum(ju, jv) < 0.004)
    h = np.where(joint, -0.003, h)
    # displaced position -> normal by finite differences
    P = np.stack([X, Y, Z], -1) + h[..., None] * n0
    du = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dv = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    du[:, 0] = du[:, 1]; du[:, -1] = du[:, -2]
    dv[0] = dv[1]; dv[-1] = dv[-2]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n = np.where((np.sum(n * n0, -1) < 0)[..., None], -n, n)
    # albedo / roughness
    rng = np.random.default_rng(5)
    from scipy import ndimage
    cloud = ndimage.gaussian_filter(rng.standard_normal((ns // 4 + 1, nx // 4 + 1)).astype(np.float32), 6)
    cloud /= np.abs(cloud).max() + 1e-6
    cloud = np.kron(cloud, np.ones((4, 4), np.float32))[:ns, :nx]
    hn = np.where(isf, (1.0 - 2.0 * tri), 1.0)
    alb_f = 0.72 * (0.90 + 0.10 * hn)                 # slight darkening in the valleys
    alb_b = np.where(joint, 0.30, 0.78)
    alb = np.where(isf, alb_f, alb_b) * (1.0 + 0.015 * cloud)
    albedo = np.stack([alb * 1.0, alb * 0.995, alb * 0.985], -1)
    rough = np.where(isf, 0.42, np.where(joint, 0.8, 0.30)) + 0.03 * cloud
    d = {"albedo": albedo.astype(np.float32), "rough": rough.astype(np.float32),
         "normal": (n * 0.5 + 0.5).astype(np.float32), "mask": isf.astype(np.float32)}
    np.savez(path, **{k: v.astype(np.float16) for k, v in d.items()})
    return d


def vault_material(V, d):
    B = H.B
    mb = B.MB("vault_real")
    uv = mb.uv("UVMap")
    ia = B.np_image("vault_albedo", d["albedo"], False)
    ir = B.np_image("vault_rough", d["rough"], True)
    inn = B.np_image("vault_normal", d["normal"], True)
    a = mb.img(ia, uv)
    r = mb.img(ir, uv)
    nn = mb.img(inn, uv)
    sep = mb.n("ShaderNodeSeparateColor")
    mb.l(r.outputs["Color"], sep.inputs["Color"])
    nm = mb.n("ShaderNodeNormalMap")
    nm.space = "OBJECT"
    mb.l(nn.outputs["Color"], nm.inputs["Color"])
    mb.l(a.outputs["Color"], mb.bsdf.inputs["Base Color"])
    mb.l(sep.outputs[0], mb.bsdf.inputs["Roughness"])
    mb.l(nm.outputs["Normal"], mb.bsdf.inputs["Normal"])
    mb.s("Specular IOR Level", 0.5)
    return mb.m


# ------------------------------------------------------------------ meshes

def vault_mesh(V, mat, step=0.07):
    nx = int(math.ceil((V.x1 - V.x0) / step))
    xs = np.linspace(V.x0, V.x1, nx + 1)
    # put grid lines on the folds between shells so the creases stay crisp
    mids = (V.cx[1:] + V.cx[:-1]) / 2
    xs = np.unique(np.concatenate([xs, mids]))
    ns = int(math.ceil(V.Stot / step))
    ss = np.linspace(0.0, V.Stot, ns + 1)
    X, Sg = np.meshgrid(xs, ss)
    Y, Z, isf, K, D = V.surface(X, Sg)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    nu = len(xs)
    rows, cols = np.meshgrid(np.arange(ns), np.arange(nu - 1), indexing="ij")
    a = rows * nu + cols
    # winding chosen so the faces point into the room (Cycles inverts object-space
    # normal maps on back faces)
    faces = np.stack([a, a + nu, a + nu + 1, a + 1], -1).reshape(-1, 4)
    me = bpy.data.meshes.new("vault")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    u = ((X - V.x0) / (V.x1 - V.x0)).ravel()
    v = (Sg / V.Stot).ravel()
    li = np.array([lp.vertex_index for lp in me.loops])
    uvl.data.foreach_set("uv", np.stack([u[li], v[li]], -1).ravel().astype(np.float32))
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new("vault", me)
    H.coll("Vault").objects.link(ob)
    return ob


def arc_strip(name, V, s0, s1, x0, x1, inset, mat, cname="Vault", nseg=40, uv_flip=False):
    """A strip following the barrel between arc lengths s0..s1, pulled 'inset' metres into the room."""
    ss = np.linspace(s0, s1, nseg + 1)
    y, z = V.at_s(ss)
    ty, tz = V.tangent_s(ss)
    y = y + tz * inset
    z = z - ty * inset
    verts = []
    for x in (x0, x1):
        verts += [(x, float(yy), float(zz)) for yy, zz in zip(y, z)]
    m = nseg + 1
    faces = [(i, m + i, m + i + 1, i + 1) for i in range(nseg)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    # orient faces into the room
    nrm = np.array([0.0, tz[nseg // 2], -ty[nseg // 2]])
    if np.dot(np.array(me.polygons[nseg // 2].normal), nrm) < 0:
        for p in me.polygons:
            p.flip()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            j, i = divmod(vi, m)
            vv = i / nseg
            uvl.data[li].uv = (float(j) if not uv_flip else 1.0 - float(j), vv)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    H.coll(cname).objects.link(ob)
    return ob


def end_walls(V, mat, z_flat, gap=None):
    """Walls closing the barrel at both ends, down to the flat ceiling; 'gap' = (y0, y1, z) leaves
    the stairwell open below z at the south end."""
    for x, nm in ((V.x0, "n"), (V.x1, "s")):
        ss = np.linspace(0, V.Stot, 121)
        y, zb = V.at_s(ss)
        zf, _, _ = V.funnel(np.full_like(y, x), y)
        z = np.minimum(zb, zf)
        pts, faces = [], []
        for i, (yy, zz) in enumerate(zip(y, z)):
            zz = max(zz, z_flat)
            zlo = z_flat
            if gap is not None and nm == "s" and gap[0] <= yy <= gap[1]:
                zlo = min(max(gap[2], z_flat), zz)
            pts += [(x, float(yy), float(zz)), (x, float(yy), zlo)]
            if i:
                a = 2 * (i - 1)
                faces.append((a, a + 2, a + 3, a + 1))
        me = bpy.data.meshes.new("vault_end_" + nm)
        me.from_pydata(pts, [], faces)
        me.update()
        H.box_uv(me)
        me.materials.append(mat)
        H.coll("Vault").objects.link(bpy.data.objects.new("vault_end_" + nm, me))


# ------------------------------------------------------------------ light boxes

def glass_texture(seed=3, N=1024):
    """Etched glass of the light boxes: a large ring, sweeping arcs and fine crackle lines."""
    rng = np.random.default_rng(seed)
    c = (np.arange(N) + 0.5) / N * 2 - 1
    U, Vv = np.meshgrid(c, c)
    R = np.hypot(U, Vv)
    A = np.arctan2(Vv, U)
    base = 0.82 + 0.18 * np.exp(-(R / 0.9) ** 2)
    lines = np.zeros_like(R)
    for r0, w, amp in ((0.62, 0.012, 0.25), (0.70, 0.006, 0.18), (0.40, 0.004, 0.12)):
        lines += amp * np.exp(-((R - r0) / w) ** 2)
    for k in range(26):       # sweeping arcs
        a0 = rng.uniform(-math.pi, math.pi)
        cxa, cya = math.cos(a0) * 1.3, math.sin(a0) * 1.3
        rr = np.hypot(U - cxa, Vv - cya)
        r1 = rng.uniform(0.7, 1.4)
        lines += 0.10 * np.exp(-((rr - r1) / 0.004) ** 2) * (R < 0.95)
    for k in range(90):       # radial strokes
        a0 = rng.uniform(-math.pi, math.pi)
        da = np.abs(np.angle(np.exp(1j * (A - a0))))
        lines += 0.06 * np.exp(-(da * R / 0.003) ** 2) * ((R > 0.45) & (R < 0.62))
    img = np.clip(base + lines, 0, 1.6)
    tint = np.array([0.93, 1.0, 0.86], np.float32)
    return (img[..., None] * tint).astype(np.float32)


def light_boxes(V, M):
    B = H.B
    gimg = B.np_image("lightbox_glass", glass_texture(), False)
    mb = B.MB("lightbox_glass")
    t = mb.img(gimg, mb.uv("UVMap"))
    mb.l(t.outputs["Color"], mb.bsdf.inputs["Base Color"])
    mb.l(t.outputs["Color"], mb.bsdf.inputs["Emission Color"])
    mb.s("Emission Strength", 0.9).s("Roughness", 0.15)
    mglass = mb.m
    hb, hh = V.box_half, V.box_h
    for k, cx in enumerate(V.cx):
        cy = V.cy
        zt, zb = V.z_top, V.z_top - hh
        # mirror-polished sides (open box), thin top plate
        for (lo, hi) in (((cx - hb, cy - hb, zb), (cx + hb, cy - hb + 0.02, zt)),
                         ((cx - hb, cy + hb - 0.02, zb), (cx + hb, cy + hb, zt)),
                         ((cx - hb, cy - hb, zb), (cx - hb + 0.02, cy + hb, zt)),
                         ((cx + hb - 0.02, cy - hb, zb), (cx + hb, cy + hb, zt))):
            H.bx(f"lb{k}_side", lo, hi, M["mirror"], "Vault", 0.004)
        H.bx(f"lb{k}_top", (cx - hb, cy - hb, zt - 0.02), (cx + hb, cy + hb, zt), M["mirror"], "Vault")
        # glass, recessed 4 cm, with a 3x3 grid of mullions and round studs
        zg = zb + 0.04
        q = B.image_quad(f"lb{k}_glass", (cx, cy, zg), (1, 0, 0), (0, -1, 0), 2 * hb - 0.04, 2 * hb - 0.04, mglass)
        H.put(q, "Vault")
        for i in range(4):
            u = -hb + 0.02 + i * (2 * hb - 0.04) / 3
            H.bx(f"lb{k}_mx{i}", (cx + u - 0.015, cy - hb, zg - 0.025), (cx + u + 0.015, cy + hb, zg), M["steel"], "Vault")
            H.bx(f"lb{k}_my{i}", (cx - hb, cy + u - 0.015, zg - 0.025), (cx + hb, cy + u + 0.015, zg), M["steel"], "Vault")
            for j in range(4):
                w = -hb + 0.02 + j * (2 * hb - 0.04) / 3
                ob = B.cylinder(f"lb{k}_stud{i}{j}", (cx + u, cy + w, zg - 0.04), 0.045, 0.03, M["steel"], segs=20)
                H.put(ob, "Vault")
        # light: down through the glass, and up onto the funnel (indirect uplight)
        L = B.area_light(f"lbA{k}", (cx, cy, zg - 0.03), 2 * hb - 0.1, 2 * hb - 0.1, 320.0, (1.0, 0.97, 0.9), camera=False)
        H.put(L, "B2_lights")
        U = B.area_light(f"lbU{k}", (cx, cy, zt + 0.05), 2 * hb, 2 * hb, 450.0, (1.0, 0.97, 0.92), rot=(math.pi, 0, 0), camera=False)
        U.data.spread = math.radians(160)
        H.put(U, "B2_lights")
        # LED lines along the top edges of each box wash the flaring shell (indirect light)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ex, ey = cx + sx * (hb + 0.05), cy + sy * (hb + 0.05)
            sz = (0.25, 2 * hb) if sx else (2 * hb, 0.25)
            E = B.area_light(f"lbE{k}_{sx}{sy}", (ex, ey, zt + 0.02), sz[0], sz[1], 140.0, (1.0, 0.97, 0.92), camera=False)
            # emit up and outward: rotate the light's -Z axis toward (sx, sy, +1)
            tilt = math.radians(130)
            if sx:
                E.rotation_euler = (0, -sx * tilt, 0)
            else:
                E.rotation_euler = (sy * tilt, 0, 0)
            H.put(E, "B2_lights")


# ------------------------------------------------------------------ downlights, LED, ad band

def downlights(V, M):
    B = H.B
    k = 0

    def disc(x, s, inset=0.012):
        nonlocal k
        y, z = V.at_s(np.array([s]))
        ty, tz = V.tangent_s(np.array([s]))
        y, z, ty, tz = float(y[0]), float(z[0]), float(ty[0]), float(tz[0])
        ang = math.atan2(tz, ty)          # rotate the disc to the local tangent
        for rad, dep, mat, nm in ((0.055, 0.012, M["steel"], "rim"), (0.040, 0.016, M["downlight"], "led")):
            ob = B.cylinder(f"dl_{nm}_{k}", (0, 0, 0), rad, dep, mat, segs=20)
            ob.location = (x, y + tz * (inset + (0.002 if nm == "led" else 0)), z - ty * (inset + (0.002 if nm == "led" else 0)))
            ob.rotation_euler = (ang, 0, 0)
            H.put(ob, "Vault")
        k += 1

    # funnel boundary along the crown line, in arc length, per side
    ys = np.linspace(V.yw, V.ye, 2000)
    s_all = V.s_of_y(ys)
    # west: transverse lines of five lights at every shell, above the ad band
    for cx in V.cx:
        y, zb = V.at_s(s_all)
        zf, _, _ = V.funnel(np.full_like(ys, cx), ys)
        inside = zf < zb
        s_fw = float(s_all[np.argmax(inside)])
        s_fe = float(s_all[len(inside) - 1 - np.argmax(inside[::-1])])
        for s in np.linspace(V.ad_s[1] + 0.6, s_fw - 0.5, 5):
            disc(cx, s)
        for s in np.linspace(s_fe + 0.5, V.led_s[0] - 0.7, 4):
            disc(cx, s)
    # continuous row along the top edge of the LED screen
    for x in np.arange(V.x0 + 0.6, V.x1 - 0.4, 0.45):
        disc(x, V.led_s[0] - 0.28)
    print("vault downlights", k)


def led_screen(V, M):
    L = 40.0
    xs0 = V.x0 + (V.x1 - V.x0 - L) / 2
    ob = arc_strip("led_vision", V, V.led_s[0], V.led_s[1], xs0, xs0 + L, 0.10, M["led_vision"], uv_flip=False)
    # the image runs along x; arc_strip puts u along x (0..1) and v along the arc
    me = ob.data
    uvl = me.uv_layers["UVMap"]
    for poly in me.polygons:
        for li in poly.loop_indices:
            u, v = uvl.data[li].uv
            uvl.data[li].uv = (u, 1.0 - v)
    # black bezel behind and around
    arc_strip("led_bezel", V, V.led_s[0] - 0.12, V.led_s[1] + 0.12, xs0 - 0.12, xs0 + L + 0.12, 0.08, M["panel_grey"])
    # LED light on the room (soft, coloured by the content in reality: use a neutral-cool fill)
    sm = (V.led_s[0] + V.led_s[1]) / 2
    y, z = V.at_s(np.array([sm]))
    ty, tz = V.tangent_s(np.array([sm]))
    nrm = np.array([0.0, float(tz[0]), -float(ty[0])])
    for x in np.arange(xs0 + 4, xs0 + L, 8.0):
        A = H.B.area_light(f"ledA_{x:.0f}", (x, float(y[0]) + nrm[1] * 0.3, float(z[0]) + nrm[2] * 0.3), 8.0, 3.6, 110.0, (0.95, 0.92, 1.0), camera=False)
        A.rotation_euler = (math.atan2(nrm[1], -nrm[2]), 0, 0)
        A.visible_glossy = False
        H.put(A, "B2_lights")


def ad_band(V, M, n_panels=6):
    xa, xb = V.x0 + 0.4, V.x1 - 0.4
    w = (xb - xa) / n_panels
    for i in range(n_panels):
        x0 = xa + i * w + 0.05
        x1 = xa + (i + 1) * w - 0.05
        ob = arc_strip(f"adband_{i}", V, V.ad_s[0], V.ad_s[1], x0, x1, 0.06, M[f"band{i % 3}"])
        me = ob.data
        uvl = me.uv_layers["UVMap"]
        # track-side wall: viewer looks toward -y, so the image runs toward -x
        for poly in me.polygons:
            for li in poly.loop_indices:
                u, v = uvl.data[li].uv
                uvl.data[li].uv = (1.0 - u, v)
    arc_strip("adband_frame", V, V.ad_s[0] - 0.08, V.ad_s[1] + 0.08, V.x0, V.x1, 0.04, M["panel_grey"])
    # cornice on top of the track wall, carrying an indirect light line
    H.bx("cornice_w", (V.x0, V.yw, V.zs - 0.18), (V.x1, V.yw + 0.32, V.zs), M["panel_white"], "Vault", 0.01)
    H.bx("cornice_w_led", (V.x0, V.yw + 0.30, V.zs - 0.02), (V.x1, V.yw + 0.33, V.zs + 0.0), M["cove"], "Vault")


def build(h, M):
    setup(h)
    V = Vault(H.PLAN, H.Z_B2)
    d = bake(V, H.T.CACHE)
    vault_mesh(V, vault_material(V, d), step=float(os.environ.get("UMEDA_VAULT_STEP", "0.07")))
    st = H.PLAN["stairs"]["J"]
    end_walls(V, M["panel_white"], H.Z_B2_CEIL, gap=(st["y0"] - 0.2, st["y1"] + 0.2, H.Z_B1 + 2.6))
    light_boxes(V, M)
    downlights(V, M)
    led_screen(V, M)
    ad_band(V, M)
    return V
