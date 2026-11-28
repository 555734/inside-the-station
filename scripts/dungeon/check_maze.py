"""Checks every module of the maze kit for objects hanging in the air (the test of
scripts/umeda/check_scene.py, module by module: a group of objects that does not touch a floor,
wall or ceiling through a chain of contacts within 2.5 cm is reported).

  python3 scripts/dungeon/check_maze.py OUT_DIR [--only I,L,...]
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "umeda"))


def tris_of(cname):
    dg = bpy.context.evaluated_depsgraph_get()
    objs, verts, tris, owner = [], [], [], []
    base = 0
    for ob in bpy.data.collections[cname].all_objects:
        if ob.type != "MESH" or ob.get("ramp"):
            continue
        me = ob.evaluated_get(dg).to_mesh()
        me.calc_loop_triangles()
        if not me.loop_triangles:
            ob.evaluated_get(dg).to_mesh_clear()
            continue
        M = ob.matrix_world
        v = np.array([tuple(M @ x.co) for x in me.vertices], np.float64)
        t = np.array([tuple(lt.vertices) for lt in me.loop_triangles], np.int64)
        ob.evaluated_get(dg).to_mesh_clear()
        objs.append(ob.name)
        verts.append(v)
        tris.append(t + base)
        owner.append(np.full(len(t), len(objs) - 1))
        base += len(v)
    return objs, np.concatenate(verts), np.concatenate(tris), np.concatenate(owner)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import kit as K
    import check_scene as CS
    only = set(a.only.split(",")) if a.only else None
    K.build_all(only=only)
    report = {}
    for name in K.MODULES:
        if only and name not in only:
            continue
        objs, V, T, own = tris_of("M_" + name)
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        print(f"== {name}: {len(objs)} objects")
        bad = CS.floating(objs, V, T, own, d)
        report[name] = bad
    json.dump(report, open(os.path.join(a.out, "floating_all.json"), "w"), ensure_ascii=False, indent=1)
    print("modules with floating groups:", {k: len(v) for k, v in report.items() if v})


if __name__ == "__main__":
    main()
