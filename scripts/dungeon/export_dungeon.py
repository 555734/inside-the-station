"""Export the maze for the walkthrough app (Godot, Android): every module once, plus the layout.

  python3 scripts/dungeon/export_dungeon.py OUT_DIR --layout LAYOUT.json [--texel 0.03] [--samples 128]
        [--raw-dir DIR] [--only I,L,...] [--no-bake]
  python3 scripts/apk/finish_station.py OUT_DIR OUT_DIR/lm_raw      # denoise into OUT/lightmaps

  1. builds every module of the kit (kit.py, kit_shops.py) around the origin,
  2. merges each module into one mesh (cut in halves where a lightmap would need more than
     --max-lm texels) with a lightmap layout at which 99 % of its surface gets --texel or finer,
  3. writes modules.glb (the chunks, in module coordinates), collision.glb (one mesh per module:
     walls, floors, furniture, ramps over the stairs), materials.json + textures/ and app.json
     (which chunks make each module, every placement of the layout in Godot coordinates, spawn
     points, reflection probe boxes),
  4. bakes each module once (Cycles, direct + indirect diffuse) with a plain corridor joined to each
     of its ports, so the light near a joint is what it is in the maze; the same lightmaps serve
     every placement of the module.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "apk"))
import kit as K            # noqa: E402
import export_station as ES  # noqa: E402  (layout, bake and material helpers of the station export)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def g(p):
    """Blender (x, y, z) -> Godot (x, z, -y)."""
    return [round(float(p[0]), 4), round(float(p[2]), 4), round(float(-p[1]), 4)]


def module_meshes(cname):
    """World-space copies (modifiers applied) of a module's meshes: what is drawn with a lightmap,
    what collides, and the goods lit per vertex (bottles, packs, books... too many and too small for
    a lightmap chart each)."""
    dg = bpy.context.evaluated_depsgraph_get()
    render, col, vlit = [], [], []
    for ob in bpy.data.collections[cname].all_objects:
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
        if ob.get("ramp"):
            col.append(me)
            continue
        if ob.get("vlm"):
            vlit.append(me)
            continue
        render.append(me)
        if not ob.get("nocol"):
            col.append(me)
    return render, col, vlit


def bake_vertex(ob, samples, raw_dir):
    """Irradiance at every face corner of a vertex-lit object (Cycles bake to a colour attribute),
    kept raw in raw_dir/<name>.npz and written encoded (sqrt(E / (E + 1)), as the lightmaps)."""
    path = os.path.join(raw_dir, ob.name + ".npz")
    me = ob.data
    if "LM" not in me.color_attributes:
        me.color_attributes.new("LM", "FLOAT_COLOR", "CORNER")
    me.color_attributes.active_color = me.color_attributes["LM"]
    n = len(me.loops)
    if os.path.exists(path):
        e = np.load(path)["e"].astype(np.float32)
    else:
        for o in bpy.context.scene.objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        sc = bpy.context.scene
        sc.cycles.samples = samples
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, target="VERTEX_COLORS", use_clear=True)
        a = np.empty(n * 4, np.float32)
        me.color_attributes["LM"].data.foreach_get("color", a)
        e = a.reshape(n, 4)[:, :3]
        os.makedirs(raw_dir, exist_ok=True)
        np.savez_compressed(path, e=e.astype(np.float16))
    e = np.clip(e, 0, None)
    enc = np.sqrt(e / (e + 1.0))
    out = np.ones((n, 4), np.float32)
    out[:, :3] = enc
    me.color_attributes["LM"].data.foreach_set("color", out.ravel())
    return float(np.percentile(e, 99)) if len(e) else 0.0


def split_lights():
    """Each module's lights into L_<module> (instanced next to the baked chunks for its bake)."""
    for name in K.MODULES:
        c = bpy.data.collections.get("M_" + name)
        if c is None:
            continue
        lc = K.BU.coll("L_" + name)
        for ob in list(c.objects):
            if ob.type == "LIGHT":
                for u in list(ob.users_collection):
                    u.objects.unlink(ob)
                lc.objects.link(ob)


def exclude_sources():
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.name.startswith(("M_", "L_")):
            lc.exclude = True


def instance(cname, pos, rot, tag):
    e = bpy.data.objects.new(f"ctx_{tag}_{cname}", None)
    e.instance_type = "COLLECTION"
    e.instance_collection = bpy.data.collections[cname]
    e.location = (pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0)
    e.rotation_euler = (0, 0, rot * math.pi / 2)
    bpy.context.scene.collection.objects.link(e)
    return e


SIDE_DIR = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def context_for(name):
    """The module's own lights, and a straight corridor (with its lights) on every port."""
    made = [instance("L_" + name, (0, 0), 0, "own")]
    for (x, y, s) in K.MODULES[name]["ports"]:
        dx, dy = SIDE_DIR[s]
        c = (x + dx * K.HC, y + dy * K.HC)
        r = 0 if s in ("N", "S") else 1
        made.append(instance("M_I", c, r, f"{x:.0f}_{y:.0f}"))
        made.append(instance("L_I", c, r, f"{x:.0f}_{y:.0f}"))
    return made


# probe boxes per module (centre, size) in module coordinates (Blender)
def module_probes(name):
    if name in ("PLAZA",):
        return [((0, 0, 2.1), (26.0, 26.0, 4.2))]
    if name == "STATION":
        out = [((0, 0, 2.0), (42.0, 26.0, 4.0))]
        for x in np.arange(-70, 71, 20.0):
            out.append(((float(x), -10.0, K.Z2 + 2.25), (20.0, 23.0, 4.5)))
        return out
    if name == "EXIT":
        return [((0, -5.5, 1.5), (6.0, 5.0, 3.0)), ((0, 3.0, 3.6), (4.4, 12.0, 6.0))]
    if name.startswith("SHOP"):
        return [((0, 0, 1.5), (6.0, 16.0, 3.0)), ((5.5, 0, 1.45), (5.0, 16.0, 2.9)), ((-5.5, 0, 1.45), (5.0, 16.0, 2.9))]
    return [((0, 0, 1.5), (16.0, 16.0, 3.0))]


def godot_probe(c, s):
    return {"pos": g(c), "size": [s[0], s[2], s[1]]}


def spawns(layout):
    """Start points: the first station's gate hall, its platform, the plaza, a shop street."""
    out = []
    ins = layout["instances"]

    def at(it, local, z, heading_local, label):
        r = it["rot"]
        x, y = local
        for _ in range(r % 4):
            x, y = -y, x
        h = math.radians(heading_local + 90 * r)
        p = (it["pos"][0] + x, it["pos"][1] + y, z)
        out.append({"name": label, "pos": g(p), "fwd": g((math.cos(h), math.sin(h), 0.0))})
    st = [i for i in ins if i["module"] == "STATION"]
    if st:
        at(st[0], (-9.5, 9.0), 0.0, -90, "あかり町駅 改札前")      # in line with a gate passage
        at(st[0], (-26.0, -14.5), K.Z2, 180, "あかり町駅 ホーム")    # on the platform, past a stair foot
    pz = [i for i in ins if i["module"] == "PLAZA"]
    if pz:
        at(pz[0], (-8.0, -11.0), 0.0, 60, "ひかり広場")
    sh = [i for i in ins if i["module"].startswith("SHOP")]
    if sh:
        at(sh[len(sh) // 2], (0.0, -7.0), 0.0, 90, "地下街 商店通り")
    if len(st) > 1:
        at(st[1], (9.5, 9.0), 0.0, -90, "もう一つの駅 改札前")
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layout", required=True)
    ap.add_argument("--texel", type=float, default=0.03)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--max-lm", type=int, default=2048)
    ap.add_argument("--max-tex", type=int, default=1024)
    ap.add_argument("--raw-dir")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-bake", action="store_true")
    ap.add_argument("--batch-px", type=float, default=6_000_000)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    raw = os.path.abspath(a.raw_dir or os.path.join(out, "lm_raw"))
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ES.BATCH_PX = a.batch_px
    layout = json.load(open(a.layout))
    used_mods = sorted({i["module"] for i in layout["instances"]} | {"I"})
    K.build_all(only=set(used_mods))
    log("kit built:", used_mods)
    # parts placed by location / rotation (gate cabinets, the office booth, rails...) get their
    # world matrix only when the view layer is evaluated: before the module collections are
    # excluded from it, or they would be exported where they were made (at the origin)
    bpy.context.view_layer.update()
    split_lights()
    exclude_sources()
    chunks, sizes, cols, mod_chunks, vobjs = [], {}, [], {}, {}
    for name in used_mods:
        render, col, vlit = module_meshes("M_" + name)
        if vlit:
            vo = ES.join_meshes(f"{name}_V", vlit)
            vobjs[name] = vo
        ob = ES.join_meshes(f"{name}_0", render)
        co = ES.join_meshes(f"COL_{name}", col)
        co.hide_render = True        # it lies on the very surfaces being baked: it would shade them black
        cols.append(co)
        for me in {id(m): m for m in render + col + vlit}.values():
            if me.users == 0:
                bpy.data.meshes.remove(me)
        splits = {}
        lay = ES.layout_chunk(ob, a.texel, a.max_lm, splits)
        mod_chunks[name] = [o.name for o, s in lay]
        for o, s in lay:
            chunks.append(o)
            sizes[o.name] = s
        log(f"{name}: {len(render)} meshes, lightmaps {[(o.name, s) for o, s in lay]}")
    # materials and textures
    used = {}
    for ob in chunks + list(vobjs.values()):
        for m in ob.data.materials:
            if m is not None:
                used[m.name] = m
    table = ES.describe_materials(list(used.values()), out, a.max_tex)
    json.dump(table, open(os.path.join(out, "materials.json"), "w"), ensure_ascii=False, indent=1)
    log("materials", len(table))
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    common = dict(export_format="GLB", use_selection=True, export_yup=True, export_apply=False)
    for k, v in (("export_image_format", "NONE"), ("export_materials", "EXPORT"), ("export_texcoords", True),
                 ("export_normals", True), ("export_all_vertex_colors", False), ("export_unused_uvs", True)):
        if k in props:
            common[k] = v

    def export(objs, path, mats=True, colors=False):
        for o in bpy.context.scene.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        kw = dict(common)
        if not mats:
            kw["export_materials"] = "NONE"
        if "export_vertex_color" in props:
            kw["export_vertex_color"] = "ACTIVE" if colors else "NONE"
        bpy.ops.export_scene.gltf(filepath=path, **kw)
    export(chunks, os.path.join(out, "modules.glb"))
    export(cols, os.path.join(out, "collision.glb"), mats=False)
    meta = {
        "modules": {n: {"chunks": mod_chunks[n], "probes": [godot_probe(c, s) for c, s in module_probes(n)]} for n in used_mods},
        "lightmaps": {ob.name: {"file": f"lightmaps/{ob.name}.png"} for ob in chunks},
        "goods": {n: vo.name for n, vo in vobjs.items()},
        "instances": [{"module": i["module"], "pos": g(i["pos"]), "rot": i["rot"], "exit": i.get("exit")} for i in layout["instances"]],
        "spawns": spawns(layout),
        "exposure": 0.55,
        "texel": a.texel, "samples": a.samples,
    }
    json.dump(meta, open(os.path.join(out, "app.json"), "w"), ensure_ascii=False, indent=1)
    log("maze data written:", len(chunks), "chunks,", sum(s * s for s in sizes.values()) / 1e6, "Mtexel")
    if a.no_bake:
        return
    K.B.render_settings(argparse.Namespace(out=os.path.join(out, "x.exr"), width=640, samples=a.samples, noise=0.01))
    bpy.context.scene.cycles.sample_clamp_indirect = 12.0
    only = set(a.only.split(",")) if a.only else set(used_mods)
    for name in used_mods:
        if name not in only:
            continue
        mine = [o for o in chunks if o.name in mod_chunks[name]]
        todo = [o for o in mine if not os.path.exists(os.path.join(raw, o.name + ".npz"))]
        vo = vobjs.get(name)
        if not todo and vo is None:
            continue
        for o in chunks:
            o.hide_render = o not in mine
        for n_, o in vobjs.items():
            o.hide_render = n_ != name
        ctx = context_for(name)
        if todo:
            ES.bake(mine, sizes, a.samples, raw)
        if vo is not None:
            p99 = bake_vertex(vo, a.samples * 4, raw)
            log(f"goods of {name} lit per vertex (p99 {p99:.2f})")
        for e in ctx:
            bpy.data.objects.remove(e)
        log(f"baked {name}")
    for o in chunks + list(vobjs.values()):
        o.hide_render = False
    if vobjs:
        export(list(vobjs.values()), os.path.join(out, "goods.glb"), colors=True)
    log("done")


if __name__ == "__main__":
    main()
