"""Checks a built station scene (.blend from build_umeda / export_station) for the faults found on
the phone: objects that hang in the air (nothing they touch leads to a floor, wall or ceiling),
and paid areas that can be left without passing a ticket gate.

  python3 scripts/umeda/check_scene.py scene.blend out_dir

Floating: every mesh object is a node; two objects are linked when a point of one (its vertices
and face centres, up to 96 of them) lies within 2.5 cm of the other's surface. Components that do
not reach the structure (floors, walls, ceilings, tunnel shells: any object with a face over 6 m2)
are reported with their names and where they are.

Gates: the B1 concourse is cut at knee, hip and chest height (0.4, 0.9, 1.4 m), walkable where a
floor is underneath and nothing stands within 0.2 m; every gate passage (the gap between two gate
cabinets) is closed for this test. From the top of every stair down to the platform (paid side) a
flood fill must not reach any exit stair foot or the far ends of the concourse (unpaid side).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PLAN = json.load(open(os.path.join(ROOT, "cache", "umeda", "plan.json")))


def scene_tris():
    dg = bpy.context.evaluated_depsgraph_get()
    objs, verts, tris, owner = [], [], [], []
    base = 0
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH" or not ob.visible_get():
            continue
        if any(c.name.endswith("_lights") or c.name == "Street_far" for c in ob.users_collection):
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


def floating(objs, V, T, own, out):
    bvh = BVHTree.FromPolygons([tuple(p) for p in V], [tuple(t) for t in T], all_triangles=True)
    a = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    area = 0.5 * np.linalg.norm(a, axis=1)
    n = len(objs)
    anchor = np.zeros(n, bool)
    np.maximum.at(anchor, own, area > 6.0)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    rng = np.random.default_rng(0)
    order = np.argsort(own, kind="stable")
    starts = np.searchsorted(own[order], np.arange(n + 1))
    for i in range(n):
        ti = order[starts[i]:starts[i + 1]]
        vi = np.unique(T[ti].ravel())
        pts = list(V[vi]) + list(V[T[ti]].mean(1))
        if len(pts) > 96:
            P = np.array(pts)
            ext = set()
            for ax in range(3):
                ext.add(int(np.argmin(P[:, ax])))
                ext.add(int(np.argmax(P[:, ax])))
            pick = list(ext) + list(rng.choice(len(P), 96 - len(ext), replace=False))
            pts = [P[k] for k in pick]
        for p in pts:
            for hit in bvh.find_nearest_range(Vector(p), 0.025):
                j = int(own[hit[2]])
                if j != i:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    # parts pushed into one another (a rod into a sign body, a trunk into its crown) touch too:
    # overlapping boxes count, between objects under 12 m (not the big floor / wall / ceiling slabs,
    # whose boxes cover everything)
    lo_ = np.full((n, 3), np.inf)
    hi_ = np.full((n, 3), -np.inf)
    P = V[T]
    np.minimum.at(lo_, own, P.min(1))
    np.maximum.at(hi_, own, P.max(1))
    small = np.where((hi_ - lo_).max(1) < 12.0)[0]
    cells = {}
    for i in small:
        for cx in range(int(lo_[i, 0] // 2), int(hi_[i, 0] // 2) + 1):
            for cy in range(int(lo_[i, 1] // 2), int(hi_[i, 1] // 2) + 1):
                cells.setdefault((cx, cy), []).append(i)
    for members in cells.values():
        for ai in range(len(members)):
            i = members[ai]
            for j in members[ai + 1:]:
                if np.all(lo_[i] < hi_[j] - 0.002) and np.all(lo_[j] < hi_[i] - 0.002):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    bad = []
    for r, members in comp.items():
        if not anchor[members].any():
            tri = np.isin(own, members)
            lo, hi = V[T[tri]].reshape(-1, 3).min(0), V[T[tri]].reshape(-1, 3).max(0)
            bad.append({"objects": sorted(objs[m] for m in members)[:12], "count": len(members),
                        "lo": lo.round(2).tolist(), "hi": hi.round(2).tolist()})
    bad.sort(key=lambda b: (b["lo"][2] < -3, b["lo"][0]))
    json.dump(bad, open(os.path.join(out, "floating.json"), "w"), ensure_ascii=False, indent=1)
    print("floating groups:", len(bad))
    for b in bad[:400]:
        print("  ", b["count"], b["objects"][:4], b["lo"], b["hi"])
    return bad


def slice_occ(V, T, zs, X0, Y0, R, W, H):
    occ = np.zeros((H, W), bool)
    for z in zs:
        s = V[T][:, :, 2] - z
        m = (s.min(1) < 0) & (s.max(1) > 0)
        tt, ss = V[T[m]], s[m]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pass
        P = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = ss[:, a], ss[:, b]
            cr = sa * sb < 0
            t = np.where(cr, sa / np.where(cr, sa - sb, 1.0), np.nan)
            P.append(tt[:, a, :2] + (tt[:, b, :2] - tt[:, a, :2]) * t[:, None])
        P = np.stack(P, 1)
        ok = ~np.isnan(P[:, :, 0])
        for k in range(len(P)):
            q = P[k][ok[k]]
            if len(q) < 2:
                continue
            A, B_ = q[0], q[1]
            L = np.linalg.norm(B_ - A)
            nn = int(L / (R * 0.5)) + 2
            pts = A + (B_ - A) * np.linspace(0, 1, nn)[:, None]
            ix = ((pts[:, 0] - X0) / R).astype(int)
            iy = ((pts[:, 1] - Y0) / R).astype(int)
            g = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            occ[iy[g], ix[g]] = True
    return occ


def gates(objs, V, T, own, out, zf=0.0):
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    X0, X1, Y0, Y1, R = -200.0, 222.0, -60.0, 50.0, 0.1
    W, H = int((X1 - X0) / R), int((Y1 - Y0) / R)
    c = V[T].mean(1)
    near = np.abs(c[:, 2] - zf) < 3.0
    Tn = T[near]
    occ = slice_occ(V, Tn, [zf + 0.4, zf + 0.9, zf + 1.4], X0, Y0, R, W, H)
    # floors at concourse level
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fl_m = (np.abs(V[T][:, :, 2] - zf).max(1) < 0.08) & (np.abs(n[:, 2]) > 0.9 * np.linalg.norm(n, axis=1))
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for t in V[T[fl_m]]:
        dr.polygon([((p[0] - X0) / R, (p[1] - Y0) / R) for p in t], fill=255)
    floor = np.array(im) > 0
    # inside a solid (a wall block drawn as a filled polygon, a column): along the floor at knee
    # height, the first face met both ways (+x and +y) is seen from behind. On a 0.3 m grid.
    bvh = BVHTree.FromPolygons([tuple(p) for p in V], [tuple(t) for t in Tn], all_triangles=True)
    k3 = 3
    solid = np.zeros_like(floor)
    ys_, xs_ = np.nonzero(floor[::k3, ::k3])
    dirs = (Vector((1, 0, 0)), Vector((0, 1, 0)))
    for iy, ix in zip(ys_, xs_):
        o = Vector((X0 + (ix * k3 + 1.5) * R, Y0 + (iy * k3 + 1.5) * R, zf + 0.6))
        inside = True
        for dv in dirs:
            hit = bvh.ray_cast(o, dv, 200.0)
            if hit[0] is None or hit[1].dot(dv) <= 0:
                inside = False
                break
        if inside:
            solid[iy * k3:(iy + 1) * k3, ix * k3:(ix + 1) * k3] = True
    floor &= ~solid
    # gate passages closed: every gate cabinet's footprint joined to its neighbours in the same line
    gim = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(gim)
    cab = {}
    names = np.array(objs)
    for i, nm in enumerate(objs):
        if nm.startswith("gate_") and nm.endswith("_0"):
            tri = own == i
            p = V[T[tri]].reshape(-1, 3)
            cab[nm] = p[:, :2].mean(0)
    pts = sorted(cab.values(), key=lambda p: (p[0], p[1]))
    used = set()
    for i, p in enumerate(pts):
        d = [np.linalg.norm(p - q) for q in pts]
        for j in np.argsort(d)[1:3]:
            if d[j] < 1.3 and (min(i, j), max(i, j)) not in used:
                used.add((min(i, j), max(i, j)))
                q = pts[j]
                gd.line([((p[0] - X0) / R, (p[1] - Y0) / R), ((q[0] - X0) / R, (q[1] - Y0) / R)], fill=255, width=3)
    gate_wall = np.array(gim) > 0
    free = floor & ~ndi.binary_dilation(occ, iterations=2) & ~gate_wall
    lab, nl = ndi.label(free)

    def cell(x, y):
        return int((y - Y0) / R), int((x - X0) / R)

    def comp_near(x, y, r=1.5):
        iy, ix = cell(x, y)
        k = int(r / R)
        win = lab[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]
        v = win[win > 0]
        return set(np.unique(v).tolist())
    paid = set()
    for k, s in PLAN["stairs"].items():
        sg = 1 if s["x_top"] < s["x_bot"] else -1
        paid |= comp_near(s["x_top"] - sg * 1.2, (s["y0"] + s["y1"]) / 2)
    unpaid = {}
    for k, e in PLAN["exits"].items():
        u = np.array(e["up"])
        foot = np.array(e["c"]) - u * (e["run"] / 2 + 1.0)
        unpaid["exit " + k] = comp_near(*foot)
    unpaid["west end"] = comp_near(-160.0, 0.0)
    unpaid["east end"] = comp_near(125.0, 0.0)
    leaks = {k: sorted(v & paid) for k, v in unpaid.items() if v & paid}
    # every exit must still be reachable: from the unpaid side of some gate (through the gates
    # the platform reaches the street)
    gate_side = set()
    for nm, p in cab.items():
        for dx in (-1.6, 1.6):
            for dy in (-1.6, 1.6):
                gate_side |= comp_near(p[0] + dx, p[1] + dy, 0.4)
    gate_side -= paid
    cut_off = [k for k, v in unpaid.items() if k.startswith("exit") and not (v & gate_side) and not (v & paid)]
    print("components: gate sides", sorted(gate_side), {k: sorted(v) for k, v in unpaid.items()})
    np.savez_compressed(os.path.join(out, "flood_lab.npz"), lab=lab.astype(np.int32), occ=occ, floor=floor)
    print("exits that cannot be reached through a gate:", cut_off if cut_off else "none")
    print("paid components:", sorted(paid))
    print("unpaid reachable from the paid side:", leaks if leaks else "none")
    paths = []
    if leaks:
        # the way out: shortest walk (0.2 m grid) from a stair top to each leaking unpaid place
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order
        k = 2
        fr = free[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).any((1, 3))
        h, w = fr.shape
        idx = -np.ones(fr.shape, np.int64)
        idx[fr] = np.arange(fr.sum())
        rows, cols = [], []
        for dy, dx in ((0, 1), (1, 0)):
            a = idx[: h - dy, : w - dx]
            b = idx[dy:, dx:]
            m = (a >= 0) & (b >= 0)
            rows += [a[m], b[m]]
            cols += [b[m], a[m]]
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(int(fr.sum()),) * 2).tocsr()
        yx = np.argwhere(fr)

        def node(x, y):
            iy, ix = int((y - Y0) / R) // k, int((x - X0) / R) // k
            best = None
            for r_ in range(0, 15):
                win = idx[max(0, iy - r_):iy + r_ + 1, max(0, ix - r_):ix + r_ + 1]
                v = win[win >= 0]
                if len(v):
                    return int(v[0])
            return best
        seeds = [node(st["x_top"] - (1 if st["x_top"] < st["x_bot"] else -1) * 1.2, (st["y0"] + st["y1"]) / 2)
                 for st in PLAN["stairs"].values()]
        for name in leaks:
            if name.startswith("exit"):
                e = PLAN["exits"][name.split()[1]]
                u = np.array(e["up"])
                tx, ty = np.array(e["c"]) - u * (e["run"] / 2 + 1.0)
            else:
                tx, ty = (-160.0, 0.0) if name == "west end" else (125.0, 0.0)
            src = node(tx, ty)
            order, pred = breadth_first_order(G, src, directed=False, return_predecessors=True)
            reach = [q for q in seeds if q is not None and (q == src or pred[q] >= 0)]
            path = []
            if reach:
                t = reach[0]
                while t >= 0 and t != src:
                    path.append(t)
                    t = pred[t]
                path = path[::-1]
            pts = [(round(X0 + (yx[q][1] + 0.5) * k * R, 1), round(Y0 + (yx[q][0] + 0.5) * k * R, 1)) for q in path[::-1]]
            paths.append((name, pts))
            print(f"  way to {name}: " + " ".join(f"({x},{y})" for x, y in pts[:: max(1, len(pts) // 40)]))
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[floor] = (60, 60, 60)
    for cc in paid:
        rgb[lab == cc] = (220, 70, 70)
    for k, v in unpaid.items():
        for cc in v - paid:
            rgb[lab == cc] = (70, 160, 220)
    rgb[occ] = (255, 255, 255)
    rgb[gate_wall] = (255, 220, 0)
    Image.fromarray(rgb[::-1]).save(os.path.join(out, "gates_flood.png"))
    for part, (xa, xb) in enumerate([(-200, -60), (-70, 70), (60, 222)]):
        Image.fromarray(rgb[::-1]).crop((int((xa - X0) / R), int((Y1 - 45) / R), int((xb - X0) / R), int((Y1 + 50) / R))).save(
            os.path.join(out, f"gates_flood_{part}.png"))
    return leaks


def main():
    blend, out = sys.argv[-2], sys.argv[-1]
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend)
    objs, V, T, own = scene_tris()
    print("objects", len(objs), "triangles", len(T))
    floating(objs, V, T, own, out)
    gates(objs, V, T, own, out)


if __name__ == "__main__":
    main()
