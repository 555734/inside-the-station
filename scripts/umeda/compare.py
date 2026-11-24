"""Overlays the official map's lines (red) on the top-down render to check the reproduction.

Output stays local (render/umeda_map_overlay.png is git-ignored) because it contains the map itself.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
r = cv2.imread(os.path.join(ROOT, "render", "umeda_top.png"))
m = cv2.imread(os.path.join(ROOT, "cache", "umeda", "map.png"))[390:1250, 280:1720]
blk = m.mean(-1) < 150
ov = r.copy()
ov[blk] = (0, 0, 230)
pad = np.full((20, m.shape[1], 3), 255, np.uint8)
out = np.vstack([m, pad, r, pad, ov])
cv2.putText(out, "official map (Osaka Metro, 2018)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(out, "3D model, top view (same scale)", (10, 860 + 20 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(out, "overlay: map lines in red on the 3D model", (10, 2 * (860 + 20) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
p = os.path.join(ROOT, "render", "umeda_map_overlay.png")
cv2.imwrite(p, out)
print(p)
