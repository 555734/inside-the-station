"""Export the Umeda model for the walkthrough app (Godot, Android).

  python3 export_station.py OUT_DIR [--samples 1024] [--texel 0.03] [--max-lm 2048] [--gpu]

Builds the same scene as build_umeda.py (without the procedural wear layers, which cannot be
exported), then:
  1. merges the ~16,000 objects per level and cuts them into 20 m chunks,
  2. gives every chunk a lightmap UV layout and bakes its lighting with Cycles
     (direct + indirect diffuse, all area lights and glowing surfaces included),
  3. writes station.glb (chunks), collision.glb (walkable surfaces, stairs as ramps),
     materials.json + textures/ (albedo / roughness / normal / emission per material),
     lightmaps/<chunk>.png and app.json (spawn points, reflection probe boxes).

Lightmaps hold irradiance E encoded as sqrt(E / (E + 1)) in 8-bit PNG; the app decodes it.
Coordinates in app.json are Godot's (x, z_blender, -y_blender).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "umeda"))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("UMEDA_EXPORT", "1")
os.environ.setdefault("UMEDA_VAULT_STEP", "0.15")

import bpy  # noqa: E402  (must come before bmesh)
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import build_umeda as BU  # noqa: E402

B = BU.B
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def g(p):
    """Blender (x, y, z) -> Godot (x, z, -y)."""
    return [round(float(p[0]), 4), round(float(p[2]), 4), round(-float(p[1]), 4)]


# ------------------------------------------------------------------ scene

def build(gpu):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ns = argparse.Namespace(out="/tmp/unused.exr", width=64, samples=1, noise=0.1)
    B.render_settings(ns)
    if gpu:
        BU.enable_gpu()
    M = BU.materials()
    BU.build_b1(M)
    BU.build_b2(M)
    BU.build_stairs(M)
    BU.build_gates(M)
    BU.build_exits(M)
    BU.build_details(M)
    BU.UP.build(BU, M)
    BU.build_b1_lights(M)
    log("scene built", len(bpy.data.objects), "objects")
    return M


def level_of(ob):
    names = [c.name for c in ob.users_collection]
    for n in names:
        if n.startswith("Stairs"):
            return "ST"
        if n.startswith("B1"):
            return "B1"
        if n.startswith("B2") or n.startswith("Vault"):
            return "B2"
    z = (ob.matrix_world @ ob.data.vertices[0].co).z if ob.data.vertices else 0.0
    return "B1" if z > -2.5 else "B2"


def is_collider(ob):
    names = [c.name for c in ob.users_collection]
    if any(n.endswith("_ceiling") or n.endswith("_lights") or n == "Vault" for n in names):
        return False
    # stair treads stay solid too; the ramps laid over them make them walkable
    return True


def baked_meshes():
    """Real, world-space copies of every mesh object (modifiers applied), grouped by level."""
    dg = bpy.context.evaluated_depsgraph_get()
    groups = {"B1": [], "B2": [], "ST": []}
    coll = []
    for ob in list(bpy.data.objects):
        if ob.type != "MESH" or not ob.data.polygons:
            continue
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        me.transform(ob.matrix_world)
        if "UVMap" not in me.uv_layers:
            if len(me.uv_layers):
                me.uv_layers[0].name = "UVMap"
            else:
                me.uv_layers.new(name="UVMap")
        for extra in [u.name for u in me.uv_layers if u.name != "UVMap"]:
            me.uv_layers.remove(me.uv_layers[extra])
        groups[level_of(ob)].append(me)
        if is_collider(ob):
            coll.append(me)
    return groups, coll


def join_meshes(name, meshes, link=True):
    """Join meshes into one object with numpy array concatenation (materials merged by name)."""
    mats, index = [], {}
    V, LV, LS, MI, UV, SM = [], [], [], [], [], []
    off_v = off_l = 0
    for me in meshes:
        nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
        if npoly == 0:
            continue
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        ls = np.empty(npoly, np.int32)
        me.polygons.foreach_get("loop_start", ls)
        mi = np.empty(npoly, np.int32)
        me.polygons.foreach_get("material_index", mi)
        sm = np.empty(npoly, bool)
        me.polygons.foreach_get("use_smooth", sm)
        uv = np.zeros(nl * 2, np.float32)
        if "UVMap" in me.uv_layers:
            me.uv_layers["UVMap"].data.foreach_get("uv", uv)
        remap = []
        for m in me.materials:
            key = m.name if m is not None else "__none__"
            if key not in index:
                index[key] = len(mats)
                mats.append(m)
            remap.append(index[key])
        remap = np.array(remap or [0], np.int32)
        V.append(co)
        LV.append(lv + off_v)
        LS.append(ls + off_l)
        MI.append(remap[np.clip(mi, 0, len(remap) - 1)])
        UV.append(uv)
        SM.append(sm)
        off_v += nv
        off_l += nl
    out = bpy.data.meshes.new(name)
    npoly = sum(len(x) for x in LS)
    out.vertices.add(off_v)
    out.vertices.foreach_set("co", np.concatenate(V))
    out.loops.add(off_l)
    out.loops.foreach_set("vertex_index", np.concatenate(LV))
    out.polygons.add(npoly)
    out.polygons.foreach_set("loop_start", np.concatenate(LS))
    out.polygons.foreach_set("material_index", np.concatenate(MI))
    out.polygons.foreach_set("use_smooth", np.concatenate(SM))
    uvl = out.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.concatenate(UV))
    out.update(calc_edges=True)
    for m in mats:
        out.materials.append(m)
    ob = bpy.data.objects.new(name, out)
    if link:
        bpy.context.scene.collection.objects.link(ob)
    return ob


def ramps_mesh():
    verts, faces = [], []
    for (y0, y1, path) in BU.RAMPS:
        for (xa, za), (xb, zb) in zip(path[:-1], path[1:]):
            k = len(verts)
            verts += [(xa, y0, za + 0.01), (xb, y0, zb + 0.01), (xb, y1, zb + 0.01), (xa, y1, za + 0.01)]
            faces.append((k, k + 1, k + 2, k + 3))
    me = bpy.data.meshes.new("ramps")
    me.from_pydata(verts, [], faces)
    me.update()
    return me


# ------------------------------------------------------------------ chunks

def chunk_key(level, x, y):
    cx = int(math.floor((x + 140.0) / 20.0))
    cy = int(math.floor((y + 60.0) / 30.0)) if level == "B1" else 0
    return f"{level}_{cx:02d}_{cy}"


def split_into_chunks(ob, level):
    """Cut the level mesh on the chunk planes, then one object per chunk."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for xs in np.arange(-140.0, 160.0, 20.0):
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(xs, 0, 0), plane_no=(1, 0, 0))
    if level == "B1":
        for ys in np.arange(-60.0, 70.0, 30.0):
            geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(0, ys, 0), plane_no=(0, 1, 0))
    bm.to_mesh(me)
    bm.free()
    n = len(me.polygons)
    cen = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("center", cen)
    cen = cen.reshape(n, 3)
    keys = np.array([chunk_key(level, c[0], c[1]) for c in cen])
    out = []
    for k in sorted(set(keys)):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        drop = [bm.faces[i] for i in np.nonzero(keys != k)[0]]
        bmesh.ops.delete(bm, geom=drop, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        cm = bpy.data.meshes.new(k)
        bm.to_mesh(cm)
        bm.free()
        for m in me.materials:
            cm.materials.append(m)
        co = bpy.data.objects.new(k, cm)
        bpy.context.scene.collection.objects.link(co)
        out.append(co)
    bpy.data.objects.remove(ob)
    return out


def surface_area(me):
    a = np.empty(len(me.polygons), np.float32)
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def lightmap_uv(ob):
    me = ob.data
    lm = me.uv_layers.new(name="Lightmap")
    me.uv_layers.active = lm
    for o in bpy.context.scene.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.select_set(False)


# ------------------------------------------------------------------ bake

def bake(chunks, sizes, samples):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.render.bake.margin = 6
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    sc.render.bake.use_pass_color = False
    sc.render.bake.use_pass_direct = True
    sc.render.bake.use_pass_indirect = True
    originals = {}
    images = {}
    for ob in chunks:
        img = bpy.data.images.new("LM_" + ob.name, sizes[ob.name], sizes[ob.name], alpha=False, float_buffer=True)
        images[ob.name] = img
        originals[ob.name] = list(ob.data.materials)
        for i, m in enumerate(ob.data.materials):
            mc = m.copy()
            nt = mc.node_tree
            node = nt.nodes.new("ShaderNodeTexImage")
            node.name = "BAKE_TARGET"
            node.image = img
            nt.nodes.active = node
            ob.data.materials[i] = mc
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for ob in chunks:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = chunks[0]
    log(f"baking {len(chunks)} chunks, {samples} samples")
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, use_clear=True, margin=6)
    log("bake done")
    for ob in chunks:          # back to the shared materials for export
        for i, m in enumerate(originals[ob.name]):
            tmp = ob.data.materials[i]
            ob.data.materials[i] = m
            bpy.data.materials.remove(tmp)
    return images


def save_lightmap(img, path):
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    e = np.clip(a.reshape(h, w, 4)[..., :3], 0, None)
    enc = np.sqrt(e / (e + 1.0))
    Image.fromarray((enc[::-1] * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(path, optimize=True)
    return float(np.percentile(e, 99))


# ------------------------------------------------------------------ materials

def _src(inp):
    return inp.links[0].from_node if inp.is_linked else None


def _uv_scale(tex):
    n = _src(tex.inputs["Vector"])
    if n is not None and n.type == "MAPPING":
        s = n.inputs["Scale"].default_value
        return [float(s[0]), float(s[1])]
    return [1.0, 1.0]


def _save_image(img, path, kind, max_size):
    """kind: srgb (linear float -> sRGB 8 bit), linear (as is), file (copy the source PNG)."""
    if kind == "file" and img.filepath:
        src = bpy.path.abspath(img.filepath)
        im = Image.open(src).convert("RGB")
    else:
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)[::-1, :, :3]
        if kind == "srgb":       # in-memory textures hold linear values
            a = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    if max(im.size) > max_size:
        s = max_size / max(im.size)
        im = im.resize((max(4, int(im.size[0] * s) // 4 * 4), max(4, int(im.size[1] * s) // 4 * 4)), Image.LANCZOS)
    im.save(path, optimize=True)


def describe_materials(used, out_dir, max_tex):
    tdir = os.path.join(out_dir, "textures")
    os.makedirs(tdir, exist_ok=True)
    saved = {}

    def tex(img, kind):
        key = (img.name, kind)
        if key not in saved:
            fn = "".join(c if c.isalnum() else "_" for c in img.name) + ("_n" if kind == "normal" else "") + ".png"
            k2 = "linear" if kind in ("normal", "linear") else kind
            if img.filepath and kind == "srgb":
                k2 = "file"
            _save_image(img, os.path.join(tdir, fn), k2, max_tex if img.size[0] < 3000 else 4096)
            saved[key] = "textures/" + fn
        return saved[key]

    table = {}
    for m in used:
        b = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
        d = {}
        if b is None:
            table[m.name] = {"color": [0.5, 0.5, 0.5], "roughness": 0.5}
            continue
        bc = b.inputs["Base Color"]
        n = _src(bc)
        if n is not None and n.type == "TEX_IMAGE" and n.image is not None:
            d["albedo"] = tex(n.image, "srgb")
            d["uv_scale"] = _uv_scale(n)
        else:
            d["color"] = [float(v) for v in bc.default_value[:3]]
        r = _src(b.inputs["Roughness"])
        if r is not None and r.type in ("SEPARATE_COLOR", "SEPRGB"):
            t = _src(r.inputs[0])
            if t is not None and t.type == "TEX_IMAGE":
                d["rough"] = tex(t.image, "linear")
        d["roughness"] = float(b.inputs["Roughness"].default_value)
        nm = _src(b.inputs["Normal"])
        if nm is not None and nm.type == "NORMAL_MAP":
            t = _src(nm.inputs["Color"])
            if t is not None and t.type == "TEX_IMAGE":
                d["normal"] = tex(t.image, "normal")
                d["normal_space"] = nm.space
        d["metallic"] = float(b.inputs["Metallic"].default_value)
        d["coat"] = float(b.inputs["Coat Weight"].default_value)
        es = float(b.inputs["Emission Strength"].default_value)
        if es > 0:
            ec = b.inputs["Emission Color"]
            n = _src(ec)
            if n is not None and n.type == "TEX_IMAGE" and n.image is not None:
                d["emission_tex"] = tex(n.image, "srgb")
            else:
                d["emission"] = [float(v) for v in ec.default_value[:3]]
            d["emission_strength"] = es
        table[m.name] = d
    return table


# ------------------------------------------------------------------ app metadata

def app_meta():
    P = BU.PLAN
    Z2, Z1 = BU.Z_B2, BU.Z_B1

    def fwd(yaw_deg):
        y = math.radians(yaw_deg)
        return g((-math.sin(y), math.cos(y), 0.0))

    st = P["stairs"]["H"]
    spawns = [
        {"name": "2番線 アーチ", "pos": g((88.5, -1.6, Z2)), "fwd": fwd(105)},
        {"name": "1番線", "pos": g((82.5, 12.6, Z2)), "fwd": fwd(94)},
        {"name": "J階段", "pos": g((96.5, -7.6, Z1)), "fwd": fwd(90)},
        {"name": "北改札内 通路", "pos": g((st["x_top"] - 6.0, -5.2, Z1)), "fwd": fwd(-90)},
        {"name": "北改札", "pos": g((-132.6, -4.8, Z1)), "fwd": fwd(-116)},
    ]
    probes = []
    px0, px1 = P["plat_x"]
    yw, ye = P["edges_y"]["lo"], P["edges_y"]["up"]
    tw, te = P["track_wall_y"]["west"], P["track_wall_y"]["east"]
    V = BU.VAULT
    for x in np.arange(px0 + 7.5, px1, 15.0):
        for (y0, y1) in ((tw, 1.3), (1.3, te)):
            under = V is not None and V.x0 < x < V.x1 and y1 <= 1.3
            ztop = (V.zc - 0.3) if under else BU.Z_B2_CEIL
            c = (x, (y0 + y1) / 2, Z2 + 1.7)
            ext = (7.5, (y1 - y0) / 2, max(ztop - Z2, 2.0))
            probes.append({"pos": g(c), "size": [ext[0] * 2, ext[2] * 2, ext[1] * 2],
                           "origin": [0, -(1.7 - (ztop - Z2) / 2), 0]})
    import cv2
    walk = [np.array(p[0], np.float32).reshape(-1, 1, 2) for p in P["b1_walk"]]
    for x in np.arange(-135, 135, 15.0):
        for y in np.arange(-45, 45, 15.0):
            if any(cv2.pointPolygonTest(w, (float(x), float(y)), False) >= 0 for w in walk):
                probes.append({"pos": g((x, y, Z1 + 1.6)), "size": [15.0, 3.6, 15.0], "origin": [0, -0.1, 0]})
    return {"spawns": spawns, "probes": probes, "exposure": 0.55}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--texel", type=float, default=0.03, help="lightmap texel size in metres")
    ap.add_argument("--max-lm", type=int, default=2048)
    ap.add_argument("--max-tex", type=int, default=1024)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "lightmaps"))

    build(a.gpu)
    groups, coll_meshes = baked_meshes()
    log("meshes", {k: len(v) for k, v in groups.items()}, "colliders", len(coll_meshes))
    # collision: everything walkable or blocking, stairs/escalators covered by ramps
    col = join_meshes("collision", coll_meshes + [ramps_mesh()])
    log("collision mesh", len(col.data.polygons), "faces")
    # remove the original objects (keep the lights for the bake)
    bpy.data.batch_remove([ob for ob in bpy.data.objects if ob.type in ("MESH", "CAMERA") and ob is not col])
    bpy.context.view_layer.update()
    log("originals removed")
    chunks = []
    for level, meshes in groups.items():
        if not meshes:
            continue
        ob = join_meshes(f"L_{level}", meshes)
        log(level, "faces", len(ob.data.polygons))
        chunks += split_into_chunks(ob, level)
    bpy.data.batch_remove([me for me in bpy.data.meshes if me.users == 0])
    log("chunks", len(chunks))
    sizes = {}
    for ob in chunks:
        area = surface_area(ob.data)
        px = math.sqrt(area / 0.55) / a.texel
        s = 256
        while s < px and s < a.max_lm:
            s *= 2
        sizes[ob.name] = s
        lightmap_uv(ob)
    log("lightmap sizes", sorted(sizes.values()))
    col.hide_render = True
    images = bake(chunks, sizes, a.samples)
    lm = {}
    for ob in chunks:
        p99 = save_lightmap(images[ob.name], os.path.join(out, "lightmaps", ob.name + ".png"))
        lm[ob.name] = {"file": f"lightmaps/{ob.name}.png", "p99": p99}
    log("lightmaps saved")
    # materials actually used
    used = {}
    for ob in chunks:
        idx = np.zeros(len(ob.data.polygons), np.int32)
        ob.data.polygons.foreach_get("material_index", idx)
        for i in set(idx.tolist()):
            m = ob.data.materials[i]
            if m is not None:
                used[m.name] = m
    table = describe_materials(list(used.values()), out, a.max_tex)
    json.dump(table, open(os.path.join(out, "materials.json"), "w"), ensure_ascii=False, indent=1)
    log("materials", len(table))
    # glb exports
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    common = dict(export_format="GLB", use_selection=True, export_yup=True, export_apply=False)
    for k, v in (("export_image_format", "NONE"), ("export_materials", "EXPORT"), ("export_texcoords", True),
                 ("export_normals", True), ("export_all_vertex_colors", False), ("export_unused_uvs", True)):
        if k in props:
            common[k] = v

    def export(objs, path, mats=True):
        for o in bpy.context.scene.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        kw = dict(common)
        if not mats:
            kw["export_materials"] = "NONE"
        bpy.ops.export_scene.gltf(filepath=path, **kw)

    col.hide_render = False
    export(chunks, os.path.join(out, "station.glb"))
    export([col], os.path.join(out, "collision.glb"), mats=False)
    meta = app_meta()
    meta["lightmaps"] = lm
    json.dump(meta, open(os.path.join(out, "app.json"), "w"), ensure_ascii=False, indent=1)
    log("done", out)


if __name__ == "__main__":
    main()
