"""Denoise baked lightmaps with Intel Open Image Denoise (the copy that ships inside the bpy wheel).

Uses OIDN's "RT" filter in HDR mode (Blender's OIDN build leaves out the "RTLightmap" weights).
No albedo/normal guides: the lightmap holds irradiance only. Runs on the CPU device so it works
anywhere bpy works.

    from lm_denoise import denoise
    clean = denoise(hdr_float32_h_w_3)
"""
import ctypes
import glob
import os

import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    cands = []
    try:
        import bpy
        root = os.path.dirname(os.path.dirname(bpy.__file__))
        cands += glob.glob(os.path.join(root, "**", "libOpenImageDenoise.so*"), recursive=True)
    except Exception:
        pass
    for d in ("/usr/local/lib/python3.13/dist-packages/bpy/lib", "/usr/local/lib/python3.13/site-packages/bpy/lib"):
        cands += glob.glob(os.path.join(d, "libOpenImageDenoise.so*"))
    cands = [c for c in cands if os.path.basename(c).startswith("libOpenImageDenoise.so")]
    if not cands:
        raise RuntimeError("libOpenImageDenoise not found next to bpy")
    lib = ctypes.CDLL(sorted(cands, key=len)[0])
    vp, cp, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
    lib.oidnNewDevice.restype = vp
    lib.oidnNewDevice.argtypes = [ctypes.c_int]
    lib.oidnCommitDevice.argtypes = [vp]
    lib.oidnGetDeviceError.restype = ctypes.c_int
    lib.oidnGetDeviceError.argtypes = [vp, ctypes.POINTER(cp)]
    lib.oidnNewFilter.restype = vp
    lib.oidnNewFilter.argtypes = [vp, cp]
    lib.oidnSetSharedFilterImage.argtypes = [vp, cp, vp, ctypes.c_int, sz, sz, sz, sz, sz]
    lib.oidnSetFilterBool.argtypes = [vp, cp, ctypes.c_bool]
    lib.oidnSetFilterInt.argtypes = [vp, cp, ctypes.c_int]
    lib.oidnCommitFilter.argtypes = [vp]
    lib.oidnExecuteFilter.argtypes = [vp]
    lib.oidnReleaseFilter.argtypes = [vp]
    lib.oidnReleaseDevice.argtypes = [vp]
    _lib = lib
    return lib


def _check(lib, dev):
    msg = ctypes.c_char_p()
    if lib.oidnGetDeviceError(dev, ctypes.byref(msg)) != 0:
        raise RuntimeError("OIDN: " + (msg.value or b"?").decode())


def denoise(hdr):
    """hdr: (h, w, 3) float32 linear irradiance. Returns the denoised array (same shape)."""
    lib = _load()
    src = np.ascontiguousarray(np.clip(hdr, 0, None), np.float32)
    out = np.empty_like(src)
    h, w = src.shape[:2]
    dev = lib.oidnNewDevice(1)                 # OIDN_DEVICE_TYPE_CPU
    lib.oidnCommitDevice(dev)
    _check(lib, dev)
    f = lib.oidnNewFilter(dev, b"RT")
    for name, arr in ((b"color", src), (b"output", out)):
        lib.oidnSetSharedFilterImage(f, name, arr.ctypes.data, 3, w, h, 0, 12, w * 12)   # FLOAT3
    lib.oidnSetFilterBool(f, b"hdr", True)
    lib.oidnSetFilterInt(f, b"quality", 6)    # OIDN_QUALITY_HIGH
    lib.oidnCommitFilter(f)
    lib.oidnExecuteFilter(f)
    _check(lib, dev)
    lib.oidnReleaseFilter(f)
    lib.oidnReleaseDevice(dev)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.ones((256, 256, 3), np.float32) * 0.5 + rng.normal(0, 0.2, (256, 256, 3)).astype(np.float32)
    o = denoise(img)
    print("noise std before", float(img.std()), "after", float(o.std()))
