"""Quick Cycles stills of modules put together (checks before any bake).

  python3 scripts/dungeon/preview.py OUT_DIR SPEC VIEW [VIEW ...] [--only I,L,...] [--samples 32]
      SPEC: placements "module@x,y,z,quarter_turns;..." (or "layout:path/layout.json")
      VIEW: name:x,y,z,heading,pitch[,lens]   (z = eye height above the floor it stands on is NOT added)
"""
import argparse
import json
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def assemble(placements):
    for k, (m, p, r) in enumerate(placements):
        e = bpy.data.objects.new(f"inst{k}_{m}", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = bpy.data.collections["M_" + m]
        e.location = p
        e.rotation_euler = (0, 0, r * math.pi / 2)
        bpy.context.scene.collection.objects.link(e)
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.name.startswith("M_"):
            lc.exclude = True


def parse_spec(spec):
    if spec.startswith("layout:"):
        L = json.load(open(spec[7:]))
        return [(i["module"], tuple(i["pos"]), i["rot"]) for i in L["instances"]]
    out = []
    for part in spec.split(";"):
        if not part.strip():
            continue
        m, rest = part.split("@")
        f = [float(v) for v in rest.split(",")]
        out.append((m, (f[0], f[1], f[2]), int(f[3])))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("spec")
    ap.add_argument("views", nargs="+")
    ap.add_argument("--only", default="")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--blend", default="")
    a = ap.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import kit as K
    import build_scene as B
    B.render_settings(argparse.Namespace(out=a.out + "/x.exr", width=a.width, samples=a.samples, noise=0.03))
    sc = bpy.context.scene
    sc.cycles.samples = a.samples
    sc.cycles.blur_glossy = 0.05
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = math.log2(0.55)
    placements = parse_spec(a.spec)
    need = sorted({p[0] for p in placements})
    K.build_all(only=set(a.only.split(",")) if a.only else set(need))
    assemble(placements)
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.blend))
    cam_d = bpy.data.cameras.new("pv")
    cam = bpy.data.objects.new("pv", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    os.makedirs(a.out, exist_ok=True)
    for v in a.views:
        name, s = v.split(":")
        f = [float(x) for x in s.split(",")]
        x, y, z, h, p = f[:5]
        cam_d.lens = f[5] if len(f) > 5 else 16
        cam.location = (x, y, z)
        cam.rotation_euler = (math.radians(90 + p), 0, math.radians(h - 90))
        sc.render.filepath = os.path.join(a.out, name + ".png")
        bpy.ops.render.render(write_still=True)
        print("rendered", name, flush=True)


if __name__ == "__main__":
    main()
