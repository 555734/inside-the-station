"""Builds the original underground corridor scene in Blender (bpy module) and renders it with Cycles.

usage: python3 build_scene.py --out ../render/test.exr --width 960 --samples 64 [--texres 2048]
"""
import argparse
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures as T  # noqa: E402
import graphics as G  # noqa: E402

# ---------------------------------------------------------------- dimensions (m)
W = 4.2
HW = W / 2
H = 2.75
Y0, Y1 = -4.0, 30.0          # main corridor extent
SY0 = 25.6                   # side corridor (turning left) spans y in [SY0, Y1]
SX1 = -9.0                   # side corridor far end
BULK_Y0, BULK_Y1, BULK_Z = 12.8, 13.4, 2.45

MACRO_FLOOR = dict(x0=SX1, x1=HW, y0=Y0, y1=Y1, px=0.01)
MACRO_WALL = dict(a0=-9.0, a1=35.0, h=H, px=0.01)


def scene():
    return bpy.context.scene


def link(ob):
    scene().collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------- images

def np_image(name, arr, noncolor):
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, 2)
    h, w = arr.shape[:2]
    rgba = np.concatenate([arr, np.ones((h, w, 1), np.float32)], -1)
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color" if noncolor else "Linear Rec.709"
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def tex_set(prefix, d):
    return {
        "albedo": np_image(prefix + "_albedo", d["albedo"], False),
        "rough": np_image(prefix + "_rough", d["rough"], True),
        "normal": np_image(prefix + "_normal", d["normal"], True),
    }


# ---------------------------------------------------------------- materials

class MB:
    def __init__(self, name):
        self.m = bpy.data.materials.new(name)
        try:
            self.m.use_nodes = True
        except Exception:
            pass
        self.nt = self.m.node_tree
        self.nt.nodes.clear()
        self.out = self.n("ShaderNodeOutputMaterial")
        self.bsdf = self.n("ShaderNodeBsdfPrincipled")
        self.l(self.bsdf.outputs["BSDF"], self.out.inputs["Surface"])

    def n(self, t, **kw):
        node = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(node, k, v)
        return node

    def l(self, a, b):
        self.nt.links.new(a, b)

    def s(self, key, val):
        self.bsdf.inputs[key].default_value = val
        return self

    def img(self, image, vec=None, interp="Linear"):
        node = self.n("ShaderNodeTexImage")
        node.image = image
        node.interpolation = interp
        if vec is not None:
            self.l(vec, node.inputs["Vector"])
        return node

    def uv(self, name):
        node = self.n("ShaderNodeUVMap")
        node.uv_map = name
        return node.outputs["UV"]


def plain(name, color, rough=0.5, metal=0.0, spec=0.5, coat=0.0):
    mb = MB(name)
    mb.s("Base Color", (*color, 1)).s("Roughness", rough).s("Metallic", metal).s("Specular IOR Level", spec)
    if coat:
        mb.s("Coat Weight", coat).s("Coat Roughness", 0.03)
    return mb.m


def tiled(name, tex, macro=None, normal_strength=1.0, rough_scale=1.0):
    mb = MB(name)
    uv = mb.uv("UVMap")
    a = mb.img(tex["albedo"], uv)
    r = mb.img(tex["rough"], uv)
    nrm = mb.img(tex["normal"], uv)
    sep = mb.n("ShaderNodeSeparateColor")
    mb.l(r.outputs["Color"], sep.inputs["Color"])
    rough = sep.outputs[0]
    if rough_scale != 1.0:
        m = mb.n("ShaderNodeMath", operation="MULTIPLY")
        mb.l(rough, m.inputs[0]); m.inputs[1].default_value = rough_scale
        rough = m.outputs[0]
    color = a.outputs["Color"]
    if macro is not None:
        mi = mb.img(macro, mb.uv("Macro"))
        ms = mb.n("ShaderNodeSeparateColor")
        mb.l(mi.outputs["Color"], ms.inputs["Color"])
        vm = mb.n("ShaderNodeVectorMath", operation="MULTIPLY")
        mb.l(color, vm.inputs[0]); mb.l(ms.outputs[0], vm.inputs[1])
        color = vm.outputs[0]
        add = mb.n("ShaderNodeMath", operation="ADD", use_clamp=True)
        mb.l(rough, add.inputs[0]); mb.l(ms.outputs[1], add.inputs[1])
        rough = add.outputs[0]
    nm = mb.n("ShaderNodeNormalMap")
    nm.uv_map = "UVMap"
    nm.inputs["Strength"].default_value = normal_strength
    mb.l(nrm.outputs["Color"], nm.inputs["Color"])
    mb.l(color, mb.bsdf.inputs["Base Color"])
    mb.l(rough, mb.bsdf.inputs["Roughness"])
    mb.l(nm.outputs["Normal"], mb.bsdf.inputs["Normal"])
    mb.s("Specular IOR Level", 0.5)
    return mb.m


def printed(name, path, rough=0.4, coat=1.0, emission=0.0, interp="Cubic"):
    mb = MB(name)
    image = bpy.data.images.load(path)
    t = mb.img(image, mb.uv("UVMap"), interp)
    mb.l(t.outputs["Color"], mb.bsdf.inputs["Base Color"])
    mb.s("Roughness", rough)
    if coat:
        mb.s("Coat Weight", coat).s("Coat Roughness", 0.02).s("Coat IOR", 1.49)
    if emission:
        mb.l(t.outputs["Color"], mb.bsdf.inputs["Emission Color"])
        mb.s("Emission Strength", emission)
    return mb.m


def emissive(name, color, strength):
    mb = MB(name)
    mb.s("Base Color", (*color, 1)).s("Emission Color", (*color, 1)).s("Emission Strength", strength)
    return mb.m


# ---------------------------------------------------------------- geometry helpers

def world_uv(axis_u, axis_v, scale, off=(0.0, 0.0)):
    def f(co):
        return ((co[axis_u] + off[0]) / scale, (co[axis_v] + off[1]) / scale)
    return f


def mesh_obj(name, verts, faces, mat, uvs=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    for lname, fn in (uvs or {}).items():
        layer = me.uv_layers.new(name=lname)
        for li, loop in enumerate(me.loops):
            layer.data[li].uv = fn[li] if isinstance(fn, list) else fn(me.vertices[loop.vertex_index].co)
    if mat:
        me.materials.append(mat)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    return link(bpy.data.objects.new(name, me))


def quad(name, pts, mat, normal, uvs=None):
    p = [Vector(x) for x in pts]
    if (p[1] - p[0]).cross(p[2] - p[0]).dot(Vector(normal)) < 0:
        pts = pts[::-1]
    return mesh_obj(name, pts, [(0, 1, 2, 3)], mat, uvs)


def image_quad(name, c, right, up, w, h, mat):
    c, r, u = Vector(c), Vector(right).normalized(), Vector(up).normalized()
    pts = [c - r * w / 2 - u * h / 2, c + r * w / 2 - u * h / 2, c + r * w / 2 + u * h / 2, c - r * w / 2 + u * h / 2]
    return mesh_obj(name, pts, [(0, 1, 2, 3)], mat, {"UVMap": [(0, 0), (1, 0), (1, 1), (0, 1)]})


def _finish_bm(name, bm, mat, bevel=0.0, segs=2, smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = (co[a], co[b])
        poly.use_smooth = smooth
    if mat:
        me.materials.append(mat)
    ob = link(bpy.data.objects.new(name, me))
    if bevel:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = segs
        m.limit_method = "ANGLE"
        try:
            m.harden_normals = False
        except Exception:
            pass
    return ob


def box(name, c, s, mat, bevel=0.0, segs=2, rot=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * s[0], v.co.y * s[1], v.co.z * s[2]))
    if rot:
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Euler(rot).to_matrix())
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(c))
    return _finish_bm(name, bm, mat, bevel, segs)


def box_mm(name, lo, hi, mat, bevel=0.0, segs=2):
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    s = [b - a for a, b in zip(lo, hi)]
    return box(name, c, s, mat, bevel, segs)


def cylinder(name, c, r, h, mat, axis="Z", segs=32, r2=None, smooth=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r,
                          radius2=r if r2 is None else r2, depth=h)
    rot = {"Z": None, "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
    if rot:
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Euler(rot).to_matrix())
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(c))
    ob = _finish_bm(name, bm, mat, smooth=False)
    for p in ob.data.polygons:
        p.use_smooth = smooth and abs(p.normal.dot(Vector({"Z": (0, 0, 1), "X": (1, 0, 0), "Y": (0, 1, 0)}[axis]))) < 0.5
    return ob


def sphere(name, c, r, mat, scale=(1, 1, 1), segs=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=segs // 2, radius=r)
    for v in bm.verts:
        v.co = Vector((v.co.x * scale[0] + c[0], v.co.y * scale[1] + c[1], v.co.z * scale[2] + c[2]))
    return _finish_bm(name, bm, mat, smooth=True)


def area_light(name, loc, sx, sy, power, color, rot=(0, 0, 0), camera=True):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = sx, sy
    ld.energy = power
    ld.color = color
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    ob.rotation_euler = rot
    ob.visible_camera = camera
    ob.visible_glossy = True
    return ob


# ---------------------------------------------------------------- scene parts

def build_shell(M):
    fm = MACRO_FLOOR
    floor_uv = {"UVMap": world_uv(0, 1, 2.4, (0.15, 0.0)),
                "Macro": lambda co: ((co[0] - fm["x0"]) / (fm["x1"] - fm["x0"]), (co[1] - fm["y0"]) / (fm["y1"] - fm["y0"]))}
    quad("floor_main", [(-HW, Y0, 0), (HW, Y0, 0), (HW, Y1, 0), (-HW, Y1, 0)], M["floor"], (0, 0, 1), floor_uv)
    quad("floor_side", [(SX1, SY0, 0), (-HW, SY0, 0), (-HW, Y1, 0), (SX1, Y1, 0)], M["floor"], (0, 0, 1), floor_uv)
    ceil_uv = {"UVMap": world_uv(0, 1, 1.2)}
    side_ceil_uv = {"UVMap": world_uv(1, 0, 1.2)}
    quad("ceil_main", [(-HW, Y0, H), (HW, Y0, H), (HW, Y1, H), (-HW, Y1, H)], M["ceiling"], (0, 0, -1), ceil_uv)
    quad("ceil_side", [(SX1, SY0, H), (-HW, SY0, H), (-HW, Y1, H), (SX1, Y1, H)], M["ceiling"], (0, 0, -1), side_ceil_uv)

    wm = MACRO_WALL
    def wall_uv(along_axis):
        return {"UVMap": world_uv(along_axis, 2, 2.4),
                "Macro": lambda co: ((co[along_axis] - wm["a0"]) / (wm["a1"] - wm["a0"]), co[2] / wm["h"])}
    # main corridor walls
    quad("wall_left", [(-HW, Y0, 0), (-HW, SY0, 0), (-HW, SY0, H), (-HW, Y0, H)], M["wall"], (1, 0, 0), wall_uv(1))
    quad("wall_right", [(HW, Y0, 0), (HW, Y1, 0), (HW, Y1, H), (HW, Y0, H)], M["wall"], (-1, 0, 0), wall_uv(1))
    quad("wall_end", [(SX1, Y1, 0), (HW, Y1, 0), (HW, Y1, H), (SX1, Y1, H)], M["wall"], (0, -1, 0), wall_uv(0))
    quad("wall_back", [(-HW, Y0, 0), (HW, Y0, 0), (HW, Y0, H), (-HW, Y0, H)], M["wall"], (0, 1, 0), wall_uv(0))
    quad("wall_side_near", [(SX1, SY0, 0), (-HW, SY0, 0), (-HW, SY0, H), (SX1, SY0, H)], M["wall"], (0, 1, 0), wall_uv(0))
    quad("wall_side_far", [(SX1, SY0, 0), (SX1, Y1, 0), (SX1, Y1, H), (SX1, SY0, H)], M["wall"], (1, 0, 0), wall_uv(1))

    # skirting (dark granite) and top trim
    sk, skd = 0.10, 0.012
    box_mm("skirt_l", (-HW, Y0, 0), (-HW + skd, SY0, sk), M["skirting"], 0.002)
    box_mm("skirt_r", (HW - skd, Y0, 0), (HW, Y1, sk), M["skirting"], 0.002)
    box_mm("skirt_end", (SX1, Y1 - skd, 0), (HW, Y1, sk), M["skirting"], 0.002)
    box_mm("skirt_sn", (SX1, SY0, 0), (-HW, SY0 + skd, sk), M["skirting"], 0.002)
    box_mm("skirt_back", (-HW, Y0, 0), (HW, Y0 + skd, sk), M["skirting"], 0.002)
    tr = 0.03
    box_mm("trim_l", (-HW, Y0, H - tr), (-HW + 0.012, SY0, H), M["alu"], 0.001)
    box_mm("trim_r", (HW - 0.012, Y0, H - tr), (HW, Y1, H), M["alu"], 0.001)
    box_mm("trim_end", (SX1, Y1 - 0.012, H - tr), (HW, Y1, H), M["alu"], 0.001)
    box_mm("trim_sn", (SX1, SY0, H - tr), (-HW, SY0 + 0.012, H), M["alu"], 0.001)
    # stainless corner guard at the turn
    box_mm("corner_guard", (-HW - 0.03, SY0 - 0.004, 0.1), (-HW + 0.004, SY0 + 0.03, 1.8), M["steel"], 0.002)
    box_mm("corner_guard2", (-HW - 0.03, SY0 - 0.004, 1.8), (-HW + 0.004, SY0 + 0.03, H - tr), M["wall_paint"], 0.002)

    # lowered bulkhead across the corridor
    box_mm("bulkhead", (-HW, BULK_Y0, BULK_Z), (HW, BULK_Y1, H), M["paint_white"], 0.004)
    # expansion joints on the walls under the bulkhead
    for x, s in ((-HW, 1), (HW, -1)):
        box_mm(f"joint_{x}", (min(x, x + s * 0.008), BULK_Y0 + 0.27, 0.1), (max(x, x + s * 0.008), BULK_Y0 + 0.33, BULK_Z), M["steel"], 0.001)


def build_tactile(M):
    # yellow base plates (texture) along main corridor and into the side corridor
    def plate(name, x0, x1, y0, y1):
        uv = {"UVMap": world_uv(0, 1, 0.6, (0.15, 0.0))}
        quad(name, [(x0, y0, 0.0015), (x1, y0, 0.0015), (x1, y1, 0.0015), (x0, y1, 0.0015)], M["tactile"], (0, 0, 1), uv)
    ys0, ys1 = -3.9, 27.6
    plate("tactile_main", -0.15, 0.15, ys0, ys1 + 0.3)
    plate("tactile_side", SX1, -0.15, ys1, ys1 + 0.3)

    # raised linear bars: one block = 4 bars, arrayed
    def bars(name, origin, along_y, count):
        bm = bmesh.new()
        for i in range(4):
            off = -0.1125 + i * 0.075
            c = (off, 0.15, 0.0045) if along_y else (0.15, off, 0.0045)
            s = (0.024, 0.265, 0.006) if along_y else (0.265, 0.024, 0.006)
            tmp = bmesh.ops.create_cube(bm, size=1.0)
            for v in tmp["verts"]:
                v.co = Vector((v.co.x * s[0] + c[0], v.co.y * s[1] + c[1], v.co.z * s[2] + c[2]))
        bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(origin))
        ob = _finish_bm(name, bm, M["tactile_bar"], bevel=0.004, segs=3)
        ob.modifiers["bevel"].limit_method = "NONE"
        arr = ob.modifiers.new("array", "ARRAY")
        arr.use_relative_offset = False
        arr.use_constant_offset = True
        arr.constant_offset_displace = (0, 0.3, 0) if along_y else (-0.3, 0, 0)
        arr.count = count
        ob.modifiers.move(1, 0)
        return ob
    n_main = int(round((ys1 - ys0) / 0.3))
    bars("tactile_bars_main", (0.0, ys0, 0.0), True, n_main)
    n_side = int(round((-0.15 - SX1) / 0.3))
    bars("tactile_bars_side", (-0.45, ys1 + 0.15, 0.0), False, n_side)
    # warning (dot) block at the corner
    bm = bmesh.new()
    for i in range(5):
        for j in range(5):
            tmp = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=0.012)
            for v in tmp["verts"]:
                v.co = Vector((v.co.x - 0.11 + i * 0.055, v.co.y + ys1 + 0.04 + j * 0.055, max(v.co.z * 0.45, 0.0) + 0.0015))
    _finish_bm("tactile_dots", bm, M["tactile_bar"], smooth=True)


def build_lights(M, power):
    col = (0.90, 0.97, 1.0)
    fix = []
    for k in range(11):
        y = -3.0 + 3.0 * k
        if BULK_Y0 - 0.7 < y < BULK_Y1 + 0.7:
            continue
        for x in (-0.95, 0.95):
            fix.append((x, y, True))
    for x in (-4.2, -7.2):
        for y in (26.75, 28.85):
            fix.append((x, y, False))
    for i, (x, y, along_y) in enumerate(fix):
        L, Wd = 1.25, 0.20
        sx, sy = (Wd, L) if along_y else (L, Wd)
        area_light(f"fl_{i}", (x, y, H - 0.013), sx, sy, power, col)
        fx, fy = sx + 0.05, sy + 0.05
        # frame ring
        t = 0.025
        box_mm(f"fx_{i}_a", (x - fx / 2, y - fy / 2, H - 0.014), (x + fx / 2, y - fy / 2 + t, H), M["fixture"], 0.002)
        box_mm(f"fx_{i}_b", (x - fx / 2, y + fy / 2 - t, H - 0.014), (x + fx / 2, y + fy / 2, H), M["fixture"], 0.002)
        box_mm(f"fx_{i}_c", (x - fx / 2, y - fy / 2, H - 0.014), (x - fx / 2 + t, y + fy / 2, H), M["fixture"], 0.002)
        box_mm(f"fx_{i}_d", (x + fx / 2 - t, y - fy / 2, H - 0.014), (x + fx / 2, y + fy / 2, H), M["fixture"], 0.002)
        box_mm(f"fx_{i}_back", (x - fx / 2, y - fy / 2, H - 0.004), (x + fx / 2, y + fy / 2, H), M["fixture_inner"])


def build_signs(M):
    # hanging exit sign
    y, z0, w, h, d = 3.3, 2.14, 1.80, 0.36, 0.10
    box_mm("sign_exit_body", (-w / 2 - 0.02, y - d / 2, z0 - 0.02), (w / 2 + 0.02, y + d / 2, z0 + h + 0.02), M["sign_body"], 0.006)
    image_quad("sign_exit_front", (0, y - d / 2 - 0.001, z0 + h / 2), (1, 0, 0), (0, 0, 1), w, h, M["sign_exit"])
    image_quad("sign_exit_back", (0, y + d / 2 + 0.001, z0 + h / 2), (-1, 0, 0), (0, 0, 1), w, h, M["sign_exit"])
    for x in (-0.7, 0.7):
        cylinder(f"sign_exit_rod{x}", (x, y, (z0 + h + H) / 2), 0.012, H - z0 - h, M["steel"])
        cylinder(f"sign_exit_cap{x}", (x, y, H - 0.01), 0.04, 0.02, M["steel"])
    # hanging transfer sign further down
    y, z0, w, h = 20.6, 2.16, 1.50, 0.30
    box_mm("sign_tr_body", (-w / 2 - 0.02, y - d / 2, z0 - 0.02), (w / 2 + 0.02, y + d / 2, z0 + h + 0.02), M["sign_body"], 0.006)
    image_quad("sign_tr_front", (0, y - d / 2 - 0.001, z0 + h / 2), (1, 0, 0), (0, 0, 1), w, h, M["sign_transfer"])
    for x in (-0.55, 0.55):
        cylinder(f"sign_tr_rod{x}", (x, y, (z0 + h + H) / 2), 0.012, H - z0 - h, M["steel"])
    # emergency guidance light on the bulkhead face
    image_quad("sign_emg", (-1.2, BULK_Y0 - 0.031, BULK_Z + 0.16), (1, 0, 0), (0, 0, 1), 0.45, 0.15, M["sign_emg"])
    box_mm("sign_emg_body", (-1.45, BULK_Y0 - 0.03, BULK_Z + 0.06), (-0.95, BULK_Y0, BULK_Z + 0.26), M["paint_white"], 0.003)
    # direction sign on the end wall
    image_quad("sign_dir", (0.2, Y1 - 0.016, 2.05), (1, 0, 0), (0, 0, 1), 1.6, 0.4, M["sign_dir"])
    box_mm("sign_dir_body", (0.2 - 0.82, Y1 - 0.015, 1.83), (0.2 + 0.82, Y1, 2.27), M["sign_body"], 0.003)


def build_posters(M):
    names = ["poster_mansion", "poster_manners", "poster_museum", "poster_ramen", "poster_travel", "poster_notice"]
    ys = [1.6, 4.25, 6.9, 9.55, 15.0, 17.65, 20.3, 22.95]
    pw, ph, zc = 0.728, 1.03, 1.47
    x = -HW + 0.022
    for i, y in enumerate(ys):
        mat = M[names[i % len(names)]]
        image_quad(f"poster_{i}", (x + 0.002, y, zc), (0, 1, 0), (0, 0, 1), pw, ph, mat)
        f, d = 0.028, 0.022
        box_mm(f"pf_{i}_t", (-HW, y - pw / 2 - f, zc + ph / 2), (x + 0.006, y + pw / 2 + f, zc + ph / 2 + f), M["alu"], 0.003)
        box_mm(f"pf_{i}_b", (-HW, y - pw / 2 - f, zc - ph / 2 - f), (x + 0.006, y + pw / 2 + f, zc - ph / 2), M["alu"], 0.003)
        box_mm(f"pf_{i}_l", (-HW, y - pw / 2 - f, zc - ph / 2), (x + 0.006, y - pw / 2, zc + ph / 2), M["alu"], 0.003)
        box_mm(f"pf_{i}_r", (-HW, y + pw / 2, zc - ph / 2), (x + 0.006, y + pw / 2 + f, zc + ph / 2), M["alu"], 0.003)
        box_mm(f"pf_{i}_back", (-HW, y - pw / 2, zc - ph / 2), (x - 0.002, y + pw / 2, zc + ph / 2), M["alu_dark"])


def build_right_wall(M):
    x = HW
    # doors
    for i, yc in enumerate((8.05, 19.4)):
        dw, dh = 0.90, 2.05
        fr = 0.05
        box_mm(f"door_{i}", (x - 0.012, yc - dw / 2, 0.0), (x, yc + dw / 2, dh), M["door"], 0.002)
        box_mm(f"door_{i}_fl", (x - 0.022, yc - dw / 2 - fr, 0.0), (x, yc - dw / 2, dh + fr), M["steel"], 0.003)
        box_mm(f"door_{i}_fr", (x - 0.022, yc + dw / 2, 0.0), (x, yc + dw / 2 + fr, dh + fr), M["steel"], 0.003)
        box_mm(f"door_{i}_ft", (x - 0.022, yc - dw / 2 - fr, dh), (x, yc + dw / 2 + fr, dh + fr), M["steel"], 0.003)
        # lever handle
        hy = yc - dw / 2 + 0.08
        cylinder(f"door_{i}_rose", (x - 0.016, hy, 1.0), 0.028, 0.008, M["steel"], axis="X")
        cylinder(f"door_{i}_neck", (x - 0.04, hy, 1.0), 0.010, 0.05, M["steel"], axis="X")
        box_mm(f"door_{i}_lever", (x - 0.072, hy - 0.0, 0.99), (x - 0.055, hy + 0.13, 1.01), M["steel"], 0.006, 3)
        cylinder(f"door_{i}_key", (x - 0.016, hy, 0.85), 0.018, 0.006, M["steel"], axis="X")
        image_quad(f"door_{i}_plate", (x - 0.0135, yc + 0.05, 1.55), (0, -1, 0), (0, 0, 1), 0.30, 0.12, M["plate_staff"])
        # kick plate
        box_mm(f"door_{i}_kick", (x - 0.014, yc - dw / 2 + 0.02, 0.0), (x - 0.011, yc + dw / 2 - 0.02, 0.25), M["steel_brushed"], 0.001)
    # ventilation louvres high on the wall
    for i, yc in enumerate((3.0, 15.9, 24.3)):
        lw, lh, zc = 0.9, 0.42, 2.30
        box_mm(f"vent_{i}_back", (x - 0.004, yc - lw / 2, zc - lh / 2), (x, yc + lw / 2, zc + lh / 2), M["vent_inside"])
        f = 0.03
        box_mm(f"vent_{i}_t", (x - 0.03, yc - lw / 2 - f, zc + lh / 2), (x, yc + lw / 2 + f, zc + lh / 2 + f), M["vent"], 0.003)
        box_mm(f"vent_{i}_b", (x - 0.03, yc - lw / 2 - f, zc - lh / 2 - f), (x, yc + lw / 2 + f, zc - lh / 2), M["vent"], 0.003)
        box_mm(f"vent_{i}_l", (x - 0.03, yc - lw / 2 - f, zc - lh / 2), (x, yc - lw / 2, zc + lh / 2), M["vent"], 0.003)
        box_mm(f"vent_{i}_r", (x - 0.03, yc + lw / 2, zc - lh / 2), (x, yc + lw / 2 + f, zc + lh / 2), M["vent"], 0.003)
        n = 9
        for k in range(n):
            z = zc - lh / 2 + (k + 0.5) * lh / n
            box(f"vent_{i}_s{k}", (x - 0.016, yc, z), (0.003, lw, lh / n * 1.25), M["vent"], 0.0, rot=(0, math.radians(-40), 0))
    # fire hydrant cabinet with indicator lamp
    yc = 11.2
    box_mm("hyd_box", (x - 0.06, yc - 0.40, 0.45), (x, yc + 0.40, 1.55), M["hyd_red"], 0.006)
    image_quad("hyd_front", (x - 0.0605, yc, 1.0), (0, -1, 0), (0, 0, 1), 0.78, 1.07, M["plate_hydrant"])
    sphere("hyd_lamp", (x - 0.02, yc, 1.72), 0.045, M["red_lamp"], scale=(1, 1, 1))
    cylinder("hyd_lamp_base", (x - 0.01, yc, 1.72), 0.06, 0.02, M["steel"], axis="X")


def build_ceiling_devices(M):
    for y in (1.5, 7.5, 16.5, 23.5):
        cylinder(f"smoke_{y}", (0.0, y, H - 0.02), 0.06, 0.04, M["white_plastic"], segs=40)
        cylinder(f"smoke_{y}_b", (0.0, y, H - 0.045), 0.04, 0.012, M["white_plastic"], segs=40)
    for y in (10.5, 25.0):
        cylinder(f"spk_{y}", (0.0, y, H - 0.006), 0.13, 0.012, M["white_plastic"], segs=48)
        cylinder(f"spk_{y}_g", (0.0, y, H - 0.013), 0.10, 0.002, M["speaker_grille"], segs=48)
    for (x, y) in ((-1.65, 9.6), (1.6, 26.6)):
        cylinder(f"cam_{y}_base", (x, y, H - 0.02), 0.09, 0.04, M["white_plastic"], segs=48)
        sphere(f"cam_{y}_dome", (x, y, H - 0.04), 0.07, M["dome"], scale=(1, 1, 0.9))


def build_hatches(M):
    for (x, y) in ((0.0, 9.0), (0.0, 18.6)):
        a, t = 0.45, 0.012
        box_mm(f"hatch_{y}_a", (x - a / 2, y - a / 2, H - 0.004), (x + a / 2, y - a / 2 + t, H), M["alu"], 0.001)
        box_mm(f"hatch_{y}_b", (x - a / 2, y + a / 2 - t, H - 0.004), (x + a / 2, y + a / 2, H), M["alu"], 0.001)
        box_mm(f"hatch_{y}_c", (x - a / 2, y - a / 2, H - 0.004), (x - a / 2 + t, y + a / 2, H), M["alu"], 0.001)
        box_mm(f"hatch_{y}_d", (x + a / 2 - t, y - a / 2, H - 0.004), (x + a / 2, y + a / 2, H), M["alu"], 0.001)


def build_camera(args):
    cam = bpy.data.cameras.new("cam")
    cam.lens = args.lens
    cam.sensor_width = 36
    cam.sensor_fit = "HORIZONTAL"
    cam.clip_start = 0.05
    cam.dof.use_dof = args.fstop > 0
    cam.dof.focus_distance = args.focus
    cam.dof.aperture_fstop = max(args.fstop, 0.1)
    cam.dof.aperture_blades = 7
    ob = link(bpy.data.objects.new("cam", cam))
    ob.location = (args.cx, args.cy, args.cz)
    ob.rotation_euler = (math.radians(90 + args.pitch), 0, math.radians(args.yaw))
    scene().camera = ob


# ---------------------------------------------------------------- main

def materials(texres):
    M = {}
    wall = tex_set("wall", T.wall_tiles(N=texres))
    floor = tex_set("floor", T.floor_tiles(N=texres))
    ceil = tex_set("ceil", T.ceiling_spandrel(N=min(texres, 2048)))
    tact = tex_set("tact", T.tactile_strip(N=1024))

    fm = MACRO_FLOOR
    nx = int((fm["x1"] - fm["x0"]) / fm["px"])
    ny = int((fm["y1"] - fm["y0"]) / fm["px"])
    xs = fm["x0"] + (np.arange(nx) + 0.5) * fm["px"]
    ys = fm["y0"] + (np.arange(ny) + 0.5) * fm["px"]
    X, Y = np.meshgrid(xs, ys)
    walk = ((np.abs(X) < HW) & (Y > Y0) & (Y < Y1)) | ((X < -HW) & (X > SX1) & (Y > SY0) & (Y < Y1))
    walk[0, :] = walk[-1, :] = False
    walk[:, 0] = walk[:, -1] = False
    fmult, fradd = T.floor_macro(walk, fm["px"])
    floor_macro = np_image("floor_macro", np.stack([fmult, fradd, np.zeros_like(fmult)], -1), True)
    wm = MACRO_WALL
    wmult, wradd = T.wall_macro(wm["a1"] - wm["a0"], wm["h"], wm["px"])
    wall_macro = np_image("wall_macro", np.stack([wmult, wradd, np.zeros_like(wmult)], -1), True)

    M["wall"] = tiled("wall", wall, wall_macro)
    M["floor"] = tiled("floor", floor, floor_macro)
    M["ceiling"] = tiled("ceiling", ceil)
    M["tactile"] = tiled("tactile", tact)
    M["tactile_bar"] = plain("tactile_bar", (0.74, 0.47, 0.03), 0.45)
    M["skirting"] = plain("skirting", (0.07, 0.07, 0.072), 0.3)
    M["alu"] = plain("alu", (0.86, 0.87, 0.88), 0.28, metal=1.0)
    M["alu_dark"] = plain("alu_dark", (0.2, 0.2, 0.2), 0.6)
    M["steel"] = plain("steel", (0.78, 0.78, 0.77), 0.22, metal=1.0)
    M["steel_brushed"] = plain("steel_brushed", (0.75, 0.75, 0.74), 0.35, metal=1.0)
    M["paint_white"] = plain("paint_white", (0.75, 0.75, 0.74), 0.45)
    M["wall_paint"] = plain("wall_paint", (0.78, 0.78, 0.77), 0.3)
    M["fixture"] = plain("fixture", (0.80, 0.80, 0.80), 0.35)
    M["fixture_inner"] = plain("fixture_inner", (0.9, 0.9, 0.9), 0.5)
    M["sign_body"] = plain("sign_body", (0.05, 0.05, 0.055), 0.35, metal=0.6)
    M["door"] = plain("door", (0.52, 0.53, 0.52), 0.42)
    M["vent"] = plain("vent", (0.58, 0.58, 0.57), 0.45)
    M["vent_inside"] = plain("vent_inside", (0.01, 0.01, 0.01), 0.9)
    M["hyd_red"] = plain("hyd_red", (0.48, 0.015, 0.02), 0.25, coat=0.5)
    M["red_lamp"] = emissive("red_lamp", (1.0, 0.04, 0.02), 25.0)
    M["white_plastic"] = plain("white_plastic", (0.82, 0.82, 0.8), 0.4)
    M["speaker_grille"] = plain("speaker_grille", (0.55, 0.55, 0.55), 0.6)
    M["dome"] = plain("dome", (0.01, 0.01, 0.012), 0.05, coat=1.0)
    gfx = G.build_all()
    for k in ("poster_mansion", "poster_manners", "poster_museum", "poster_ramen", "poster_travel", "poster_notice"):
        M[k] = printed(k, gfx[k], rough=0.45, coat=1.0)
    M["sign_exit"] = printed("sign_exit", gfx["sign_exit"], rough=0.3, coat=0.0, emission=1.3)
    M["sign_transfer"] = printed("sign_transfer", gfx["sign_transfer"], rough=0.3, coat=0.0, emission=1.6)
    M["sign_emg"] = printed("sign_emg", gfx["sign_emergency"], rough=0.3, coat=0.0, emission=1.6)
    M["sign_dir"] = printed("sign_dir", gfx["wall_direction"], rough=0.3, coat=0.5, emission=0.0)
    M["plate_staff"] = printed("plate_staff", gfx["plate_staff"], rough=0.3, coat=0.5)
    M["plate_hydrant"] = printed("plate_hydrant", gfx["plate_hydrant"], rough=0.25, coat=0.6)
    return M


def render_settings(args):
    sc = scene()
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = args.noise
    cy.max_bounces = 10
    cy.diffuse_bounces = 6
    cy.glossy_bounces = 6
    cy.transmission_bounces = 2
    cy.transparent_max_bounces = 4
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    cy.blur_glossy = 0.4
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        cy.denoising_prefilter = "ACCURATE"
    except Exception as e:
        print("denoiser setup:", e)
    sc.render.resolution_x = args.width
    sc.render.resolution_y = int(round(args.width * 9 / 16))
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.filepath = os.path.abspath(args.out)
    world = bpy.data.worlds.new("world")
    sc.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0
    else:
        world.color = (0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../render/test.exr")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--texres", type=int, default=4096)
    ap.add_argument("--power", type=float, default=40.0)
    ap.add_argument("--lens", type=float, default=20.0)
    ap.add_argument("--cx", type=float, default=0.25)
    ap.add_argument("--cy", type=float, default=-1.2)
    ap.add_argument("--cz", type=float, default=1.60)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--focus", type=float, default=9.0)
    ap.add_argument("--fstop", type=float, default=1.1)
    ap.add_argument("--blend", default="")
    args = ap.parse_args()

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    render_settings(args)
    M = materials(args.texres)
    print(f"materials ready {time.time() - t0:.1f}s")
    build_shell(M)
    build_tactile(M)
    build_lights(M, args.power)
    build_signs(M)
    build_posters(M)
    build_right_wall(M)
    build_ceiling_devices(M)
    build_hatches(M)
    build_camera(args)
    print(f"scene built {time.time() - t0:.1f}s, objects={len(bpy.data.objects)}")
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"render done {time.time() - t1:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
