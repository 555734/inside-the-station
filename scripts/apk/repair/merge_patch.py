"""Put a patch bake into a full bake.
  python3 merge_patch.py CHUNK BASE.npz PATCHDIR OUT.npz [STATION.glb]
PATCHDIR/<chunk>_sliver_mask.npy: texels the base has wrong (old and new places of the patched
charts); PATCHDIR/<chunk>_target_mask.npy: the patched charts in the app's layout.
Inside that area (2 px wider): texels of patched charts and their 6 px bake margin come from the
patch (zeros included: a chart can be dark); texels of other charts stay; gutter texels within
6 px of a chart take the nearest chart texel (as the bake margin does); the rest is cleared."""
import json, struct, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
CHUNK, BASE, PD, OUT = sys.argv[1:5]
GLB = sys.argv[5] if len(sys.argv) > 5 else "app/station/station.glb"
base = np.load(BASE)["lm"].astype(np.float32)
patch = np.load(f"{PD}/{CHUNK}.npz")["lm"].astype(np.float32)
R = ndimage.binary_dilation(np.load(f"{PD}/{CHUNK}_sliver_mask.npy"), iterations=2)
T = np.load(f"{PD}/{CHUNK}_target_mask.npy")
s = base.shape[0]
f = open(GLB, "rb").read()
L = struct.unpack("<I", f[12:16])[0]
js = json.loads(f[20:20 + L]); B = 20 + L + 8
def acc(i):
    a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
    off = B + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = np.frombuffer(f, dt, a["count"] * n, off)
    return arr.reshape(-1, n) if n > 1 else arr
node = next(n for n in js["nodes"] if n.get("name") == CHUNK)
im = Image.new("L", (s, s), 0); d = ImageDraw.Draw(im)
for p in js["meshes"][node["mesh"]]["primitives"]:
    uv = acc(p["attributes"]["TEXCOORD_1"]).astype(np.float64)
    idx = acc(p["indices"]).reshape(-1, 3)
    for t in idx:
        d.polygon([(uv[k][0] * s, (1 - uv[k][1]) * s) for k in t], fill=255)
A = np.asarray(im) > 0                                   # every chart, app layout (rows bottom-up)
others = A & ~T
T6 = ndimage.binary_dilation(T, iterations=6) & ~others
out = base.copy()
use = R & T6
out[use] = patch[use]
rest = R & ~T6 & ~others
near = rest & ndimage.binary_dilation(A, iterations=6)
valid = ~R | use | (R & others)
_, ind = ndimage.distance_transform_edt(~valid, return_indices=True)
out[near] = out[ind[0][near], ind[1][near]]
out[rest & ~near] = 0
np.savez_compressed(OUT, lm=out.astype(np.float16))
print(CHUNK, "area", int(R.sum()), "from patch", int(use.sum()), "other charts kept", int((R & others).sum()),
      "margin-filled", int(near.sum()), "cleared", int((rest & ~near).sum()),
      "changed texels", int((np.abs(out - base).max(-1) > 1e-3).sum()))
