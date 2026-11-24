"""Procedural PBR textures for the corridor (numpy only).

All arrays are float32, row 0 = bottom (Blender image convention).
Height maps are in metres so normals are physically scaled.
"""
import os
import numpy as np
from scipy import ndimage

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")


def _rng(seed):
    return np.random.default_rng(seed)


def fbm(shape, sigmas, weights, seed, wrap=True):
    """Tileable band-limited noise: sum of blurred white noise, normalised to ~[-1,1]."""
    rng = _rng(seed)
    out = np.zeros(shape, np.float32)
    mode = "wrap" if wrap else "reflect"
    for s, w in zip(sigmas, weights):
        n = rng.standard_normal(shape).astype(np.float32)
        n = ndimage.gaussian_filter(n, s, mode=mode)
        n /= (n.std() + 1e-8)
        out += w * n
    out /= (np.abs(out).max() + 1e-8)
    return out


def height_to_normal(h, px_m, strength=1.0):
    """Tangent-space normal map (OpenGL, +Y up) from a height map in metres."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * px_m)
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2 * px_m)
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / l * 0.5 + 0.5, ny / l * 0.5 + 0.5, nz / l * 0.5 + 0.5], -1).astype(np.float32)


def tile_layout(N, size_m, tw, th, grout, seed, bond=0.0):
    """Returns per-pixel tile id, local coords (m) and distance to tile edge (m)."""
    px = size_m / N
    c = (np.arange(N, dtype=np.float32) + 0.5) * px
    U, V = np.meshgrid(c, c)  # V rows (bottom->top)
    row = np.floor(V / th).astype(np.int32)
    Ushift = U + (row % 2) * bond * tw
    col = np.floor(Ushift / tw).astype(np.int32)
    lu = Ushift - col * tw
    lv = V - row * th
    ncol = int(round(size_m / tw))
    nrow = int(round(size_m / th))
    tid = (row % nrow) * 1000 + (col % ncol)
    de = np.minimum(np.minimum(lu, tw - lu), np.minimum(lv, th - lv)) - grout / 2
    return tid, lu, lv, de, row % nrow, col % ncol, px


def per_tile(tid, seed, scale=1.0):
    ids, inv = np.unique(tid, return_inverse=True)
    r = _rng(seed).standard_normal(len(ids)).astype(np.float32) * scale
    return r[inv].reshape(tid.shape)


def wall_tiles(N=4096, size_m=2.4, tile=0.15, grout=0.0055, stripe_row=7, seed=11, tag=""):
    path = os.path.join(CACHE, f"wall{tag}_{N}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k].astype(np.float32) for k in d}
    tid, lu, lv, de, row, col, px = tile_layout(N, size_m, tile, tile, grout, seed)
    r_edge = 0.0026
    t = np.clip(de / r_edge, 0, 1)
    prof = np.sqrt(1 - (1 - t) ** 2)            # rounded "cushion" edge
    is_tile = de > 0
    h = prof * 0.0025
    # slight tilt and convexity per tile -> broken reflections like real glazed tiles
    sx = per_tile(tid, seed + 1, 0.0016)
    sy = per_tile(tid, seed + 2, 0.0016)
    h += is_tile * (sx * (lu - tile / 2) + sy * (lv - tile / 2))
    h += is_tile * (-0.010 * ((lu - tile / 2) ** 2 + (lv - tile / 2) ** 2))
    ripple = fbm((N, N), [6, 18, 60], [0.35, 0.45, 0.6], seed + 3)
    h += is_tile * ripple * 0.000020
    grout_n = fbm((N, N), [1.0, 3.0], [0.6, 0.4], seed + 4)
    h += (~is_tile) * (-0.0006 + grout_n * 0.00012)
    normal = height_to_normal(h, px)

    # albedo
    v = per_tile(tid, seed + 5, 1.0)
    base = 0.80 + 0.012 * v
    warm = per_tile(tid, seed + 6, 0.006)
    alb = np.stack([base + warm, base + warm * 0.3, base - warm * 0.6], -1)
    # glaze micro speckle
    spk = fbm((N, N), [0.7], [1.0], seed + 7)
    alb *= (1 - 0.015 * np.clip(spk, 0, 1))[..., None]
    if stripe_row is not None:
        s = (row == stripe_row)[..., None] & is_tile[..., None]
        line = np.array([0.020, 0.135, 0.205], np.float32)  # line colour (muted blue)
        alb = np.where(s, line * (1 + 0.04 * v[..., None]), alb)
    gcol = 0.085 + 0.02 * grout_n
    gdark = np.clip(1 - t, 0, 1)  # dirt collects at the very edge of the glaze
    alb = np.where(is_tile[..., None], alb * (1 - 0.05 * gdark[..., None] ** 3), gcol[..., None] * np.array([1.0, 0.99, 0.96]))
    # roughness
    smudge = fbm((N, N), [40, 120], [0.5, 0.5], seed + 8)
    rough = 0.055 + 0.02 * np.abs(per_tile(tid, seed + 9, 1.0)) + 0.06 * np.clip(smudge, 0, 1)
    rough = np.where(is_tile, rough + 0.25 * (1 - t) ** 4, 0.85)
    out = {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": normal}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **{k: v.astype(np.float16) for k, v in out.items()})
    return out


def floor_tiles(N=4096, size_m=2.4, tile=0.30, grout=0.0035, seed=23):
    path = os.path.join(CACHE, f"floor_{N}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k].astype(np.float32) for k in d}
    tid, lu, lv, de, row, col, px = tile_layout(N, size_m, tile, tile, grout, seed)
    is_tile = de > 0
    t = np.clip(de / 0.0015, 0, 1)
    h = np.sqrt(1 - (1 - t) ** 2) * 0.0015
    sx = per_tile(tid, seed + 1, 0.0009)
    sy = per_tile(tid, seed + 2, 0.0009)
    h += is_tile * (sx * (lu - tile / 2) + sy * (lv - tile / 2))
    grain = fbm((N, N), [0.8, 2.5], [0.5, 0.5], seed + 3)
    h += is_tile * grain * 0.000015
    h += (~is_tile) * (-0.0012)
    normal = height_to_normal(h, px)
    # porcelain stoneware with granite-like speckle
    v = per_tile(tid, seed + 4, 1.0)
    base = 0.50 + 0.015 * v
    rng = _rng(seed + 5)
    dots = rng.random((N, N)).astype(np.float32)
    dark = ndimage.gaussian_filter((dots < 0.018).astype(np.float32), 1.1, mode="wrap")
    light = ndimage.gaussian_filter((dots > 0.985).astype(np.float32), 1.0, mode="wrap")
    cloud = fbm((N, N), [10, 40], [0.5, 0.5], seed + 6)
    lum = base * (1 + 0.04 * cloud) - 1.6 * dark * base + 1.2 * light * 0.15
    lum = np.clip(lum, 0.05, 0.9)
    alb = np.stack([lum * 1.0, lum * 0.995, lum * 0.975], -1)
    gcol = 0.16 + 0.03 * grain
    alb = np.where(is_tile[..., None], alb, gcol[..., None] * np.array([1.0, 0.97, 0.92]))
    scuff = fbm((N, N), [25, 90], [0.5, 0.5], seed + 7)
    rough = 0.13 + 0.05 * np.clip(scuff, -1, 1) + 0.02 * np.abs(v)
    rough = np.where(is_tile, rough + 0.25 * (1 - t) ** 3, 0.8)
    out = {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": normal}
    np.savez(path, **{k: v.astype(np.float16) for k, v in out.items()})
    return out


def tactile_strip(N=1024, size_m=0.6, tile=0.30, seed=31):
    """Yellow tactile blocks (base plate texture; the raised bars are geometry)."""
    tid, lu, lv, de, row, col, px = tile_layout(N, size_m, tile, tile, 0.004, seed)
    is_tile = de > 0
    t = np.clip(de / 0.002, 0, 1)
    h = np.sqrt(1 - (1 - t) ** 2) * 0.001 + (~is_tile) * -0.001
    grain = fbm((N, N), [1.0, 4.0], [0.5, 0.5], seed + 1)
    h += grain * 0.00002
    normal = height_to_normal(h, px)
    v = per_tile(tid, seed + 2, 1.0)
    y = np.array([0.78, 0.50, 0.035], np.float32)
    alb = y * (1 + 0.03 * v[..., None]) * (1 + 0.05 * grain[..., None])
    alb = np.where(is_tile[..., None], alb, np.array([0.12, 0.11, 0.09], np.float32))
    rough = np.where(is_tile, 0.42 + 0.05 * grain, 0.8)
    return {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": normal}


def ceiling_spandrel(N=2048, size_m=1.2, pitch=0.10, groove=0.006, seed=41):
    """Painted aluminium spandrel: strips run along V."""
    px = size_m / N
    c = (np.arange(N, dtype=np.float32) + 0.5) * px
    U, V = np.meshgrid(c, c)
    lu = np.mod(U, pitch)
    d = np.minimum(lu, pitch - lu) - groove / 2
    t = np.clip(d / 0.003, 0, 1)
    h = np.sqrt(1 - (1 - t) ** 2) * 0.004
    wav = fbm((N, N), [30, 100], [0.5, 0.5], seed)
    h += wav * 0.00003
    normal = height_to_normal(h, px)
    lum = 0.74 + 0.01 * wav
    groove_dark = (d <= 0)
    lum = np.where(groove_dark, 0.45, lum * (1 - 0.05 * (1 - t) ** 2))
    alb = np.stack([lum, lum * 0.995, lum * 0.985], -1)
    rough = np.where(groove_dark, 0.8, 0.34 + 0.04 * wav)
    return {"albedo": alb.astype(np.float32), "rough": rough.astype(np.float32), "normal": normal}


# ---------------- macro (non repeating) dirt / wear maps -----------------

def floor_macro(walk_mask, px_m, seed=51):
    """walk_mask: bool array (rows=y, cols=x) of floor area. Returns albedo mult & roughness add."""
    shp = walk_mask.shape
    dist = ndimage.distance_transform_edt(walk_mask) * px_m
    edge = np.exp(-dist / 0.12)
    blot = fbm(shp, [8, 30, 90], [0.3, 0.4, 0.5], seed, wrap=False)
    lane = np.exp(-dist / 0.6)
    mult = 1.0 - 0.28 * edge - 0.06 * np.clip(blot, 0, 1) - 0.04 * lane
    radd = 0.10 * edge + 0.10 * np.clip(blot, -0.3, 1) - 0.03 * (1 - lane)
    # rubber scuff marks
    rng = _rng(seed + 1)
    marks = np.zeros(shp, np.float32)
    H, W = shp
    for _ in range(int(H * W * px_m * px_m * 0.25)):
        y, x = rng.integers(0, H), rng.integers(0, W)
        if not walk_mask[y, x] or dist[y, x] < 0.2:
            continue
        L = rng.integers(3, 14)
        ang = rng.uniform(0, np.pi)
        for k in range(L):
            yy = int(y + k * np.sin(ang)); xx = int(x + k * np.cos(ang))
            if 0 <= yy < H and 0 <= xx < W:
                marks[yy, xx] = max(marks[yy, xx], rng.uniform(0.3, 1.0))
    marks = ndimage.gaussian_filter(marks, 0.7)
    mult *= 1 - 0.35 * np.clip(marks, 0, 1)
    radd += 0.15 * np.clip(marks, 0, 1)
    return mult.astype(np.float32), radd.astype(np.float32)


def wall_macro(length_m, height_m, px_m, seed=61):
    W = int(length_m / px_m); H = int(height_m / px_m)
    z = (np.arange(H, dtype=np.float32) + 0.5)[:, None] * px_m
    z = np.repeat(z, W, 1)
    splash = np.exp(-np.maximum(z - 0.10, 0) / 0.06) * (z > 0.0)
    blot = fbm((H, W), [6, 25], [0.5, 0.5], seed, wrap=False)
    hand = np.exp(-((z - 1.05) / 0.25) ** 2)
    hand_n = np.clip(fbm((H, W), [4, 12], [0.6, 0.4], seed + 1, wrap=False), 0, 1)
    streak = ndimage.gaussian_filter(_rng(seed + 2).standard_normal((H, W)).astype(np.float32), (40, 1.2))
    streak = np.clip(streak / (streak.std() + 1e-6) - 1.6, 0, 1) * (z < 2.0)
    mult = 1.0 - 0.22 * splash * (0.7 + 0.3 * np.clip(blot, -1, 1)) - 0.035 * streak - 0.02 * np.clip(blot, 0, 1)
    radd = 0.10 * hand * hand_n + 0.10 * splash + 0.03 * streak
    return mult.astype(np.float32), radd.astype(np.float32)
