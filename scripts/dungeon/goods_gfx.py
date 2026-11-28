"""Packaging of the goods in the maze's shops (drawn with Pillow): one atlas of 8 x 8 cells for
labels and fronts, one for book spines and covers. All brands and products are made up.

  goods.png (4096 x 4096): row 0 bottle labels, 1 can labels, 2 snack bags, 3 boxes, 4 cup noodles
      (sides, lids), 5 rice balls / sandwiches / lunch boxes, 6 drug store goods, 7 magazines,
      colour swatches, bread crust
  books.png (2048 x 2048): 64 spines (32 x 2 rows of 64 x 512 px), 8 covers (256 x 512)
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from graphics import F, text  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "cache", "dungeon", "gfx")
CELL = 512
N = 8
SWATCH_CELL = (4, 7)
SWATCHES = [  # (name, colour) on an 8 x 8 grid inside the swatch cell
    ("white", (238, 238, 234)), ("cap_white", (245, 245, 245)), ("cap_blue", (40, 90, 190)), ("cap_red", (200, 30, 30)),
    ("cap_green", (30, 140, 70)), ("cap_orange", (240, 140, 30)), ("cap_yellow", (240, 200, 30)), ("black", (25, 25, 25)),
    ("can_top", (190, 192, 196)), ("clear_water", (200, 220, 230)), ("tea_green", (150, 160, 60)), ("tea_brown", (120, 70, 30)),
    ("cola", (40, 20, 15)), ("soda_yellow", (230, 220, 120)), ("milk_tea", (200, 160, 110)), ("orange_juice", (240, 150, 40)),
    ("sports", (190, 220, 235)), ("paper", (240, 236, 224)), ("cardboard", (180, 140, 95)), ("rice", (245, 243, 236)),
    ("seaweed", (30, 40, 30)), ("pink", (240, 150, 170)), ("cream", (245, 230, 200)), ("brown", (110, 70, 40)),
    ("bread_light", (215, 165, 95)), ("bread_dark", (165, 100, 45)), ("chocolate", (80, 45, 25)), ("strawberry", (210, 40, 60)),
    ("cup_white", (240, 240, 238)), ("coffee", (70, 40, 20)), ("leaf", (50, 120, 50)), ("stem", (70, 130, 60)),
    ("petal_red", (200, 30, 60)), ("petal_yellow", (245, 200, 40)), ("petal_pink", (240, 140, 180)), ("petal_white", (245, 245, 240)),
    ("petal_purple", (130, 60, 160)), ("wrap_kraft", (200, 170, 120)), ("wrap_clear", (230, 235, 235)), ("plastic_tray", (60, 60, 62)),
    ("bowl_red", (170, 30, 25)), ("bowl_black", (30, 28, 26)), ("broth", (190, 140, 70)), ("noodle", (235, 210, 140)),
    ("fabric_navy", (30, 40, 70)), ("fabric_grey", (140, 140, 140)), ("fabric_beige", (200, 185, 160)), ("fabric_white", (235, 235, 230)),
    ("fabric_black", (30, 30, 32)), ("fabric_green", (60, 90, 70)), ("fabric_red", (150, 40, 40)), ("denim", (50, 70, 110)),
    ("steel", (180, 182, 186)), ("price_yellow", (250, 220, 60)), ("glass_green", (90, 140, 100)), ("label_red", (210, 40, 40)),
    ("magazine_edge", (220, 220, 215)), ("book_page", (235, 230, 215)), ("shampoo_pump", (240, 240, 240)), ("cosmetic_gold", (200, 170, 100)),
    ("ice", (220, 235, 245)), ("plum", (150, 30, 50)), ("salmon", (240, 130, 90)), ("tuna", (220, 200, 160)),
]


def cell_uv(col, row, atlas=N):
    """(u0, v0, u1, v1) of a cell in Blender UV space (v up; row 0 at the top of the image)."""
    u0, u1 = col / atlas, (col + 1) / atlas
    v1 = 1.0 - row / atlas
    v0 = 1.0 - (row + 1) / atlas
    pad = 2.0 / (CELL * atlas)
    return (u0 + pad, v0 + pad, u1 - pad, v1 - pad)


def swatch_uv(name):
    i = [s[0] for s in SWATCHES].index(name)
    c, r = SWATCH_CELL
    u0, v0, u1, v1 = cell_uv(c, r)
    gx, gy = i % 8, i // 8
    du, dv = (u1 - u0) / 8, (v1 - v0) / 8
    cu = u0 + du * (gx + 0.5)
    cv = v1 - dv * (gy + 0.5)
    e = min(du, dv) * 0.2
    return (cu - e, cv - e, cu + e, cv + e)


def _rng(seed):
    return np.random.default_rng(seed)


def fit(d, xy, s, font, size, width, fill, anchor="mm"):
    """Text in the given font, as large as size but no wider than width."""
    while size > 12 and d.textlength(s, font=F(font, size)) > width:
        size -= 2
    text(d, xy, s, F(font, size), fill, anchor)


def _fine(d, x, y, w, lines, color, size=16):
    f = F("sans", size)
    for i in range(lines):
        d.rectangle([x, y + i * (size + 6), x + w * (0.6 + 0.4 * ((i * 37) % 10) / 10), y + i * (size + 6) + size * 0.55], fill=color)


def _barcode(d, x, y, w, h, seed):
    r = _rng(seed)
    d.rectangle([x - 6, y - 6, x + w + 6, y + h + 22], fill=(255, 255, 255))
    xx = x
    while xx < x + w:
        bw = int(r.integers(1, 4))
        if r.random() < 0.55:
            d.rectangle([xx, y, xx + bw - 1, y + h], fill=(10, 10, 10))
        xx += bw
    text(d, (x + w / 2, y + h + 12), "4 901234 5678" + str(seed % 10), F("sans", 14), (10, 10, 10), "mm")


def _grad(w, h, a, b, horizontal=False):
    t = np.linspace(0, 1, w if horizontal else h)
    g = (np.array(a)[None] * (1 - t[:, None]) + np.array(b)[None] * t[:, None]).astype(np.uint8)
    if horizontal:
        arr = np.repeat(g[None], h, 0)
    else:
        arr = np.repeat(g[:, None], w, 1)
    return Image.fromarray(arr, "RGB")


def bottle_label(name, sub, bg, fg, accent, motif, seed):
    """Wrap-around label of a PET bottle (the whole cell goes once round the bottle)."""
    im = _grad(CELL, CELL, bg, tuple(int(c * 0.85) for c in bg))
    d = ImageDraw.Draw(im)
    W = CELL
    # front (middle half): name, motif; back: fine print and barcode
    if motif == "mountain":
        d.polygon([(W * 0.25, 360), (W * 0.42, 240), (W * 0.55, 330), (W * 0.65, 260), (W * 0.75, 360)], fill=accent)
        d.polygon([(W * 0.39, 262), (W * 0.42, 240), (W * 0.45, 262)], fill=(255, 255, 255))
    elif motif == "leaf":
        for k in range(3):
            cx, cy = W * (0.38 + 0.12 * k), 330 - 20 * (k % 2)
            d.ellipse([cx - 30, cy - 55, cx + 30, cy + 55], fill=accent)
            d.line([cx, cy - 50, cx, cy + 50], fill=tuple(int(c * 0.7) for c in accent), width=3)
    elif motif == "fruit":
        for k in range(2):
            cx = W * (0.42 + 0.16 * k)
            d.ellipse([cx - 48, 270, cx + 48, 366], fill=accent)
            d.ellipse([cx - 20, 288, cx + 2, 310], fill=(255, 255, 255))
        d.ellipse([W * 0.47, 250, W * 0.53, 280], fill=(60, 140, 60))
    elif motif == "bubbles":
        r = _rng(seed)
        for _ in range(26):
            cx, cy, rr = r.uniform(W * 0.25, W * 0.75), r.uniform(220, 400), r.uniform(5, 18)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=accent, width=3)
    elif motif == "cup":
        d.rounded_rectangle([W * 0.40, 260, W * 0.60, 370], 18, fill=accent)
        d.arc([W * 0.58, 280, W * 0.66, 340], -90, 90, fill=accent, width=8)
    d.rectangle([W * 0.25, 90, W * 0.75, 96], fill=accent)
    fit(d, (W / 2, 160), name, "maruK", 64, W * 0.5, fg)
    text(d, (W / 2, 210), sub, F("sansM", 22), fg, "mm")
    text(d, (W / 2, 430), "500ml", F("sansB", 26), fg, "mm")
    _fine(d, 14, 70, W * 0.2, 10, tuple(int(c * 0.6) for c in fg), 12)
    _barcode(d, W * 0.80, 300, 80, 60, seed)
    return im


def can_label(name, sub, bg, fg, accent, seed):
    im = _grad(CELL, CELL, bg, tuple(min(255, int(c * 1.15)) for c in bg), horizontal=True)
    d = ImageDraw.Draw(im)
    W = CELL
    d.rectangle([0, 0, W, 26], fill=(190, 192, 196))
    d.rectangle([0, CELL - 26, W, CELL], fill=(170, 172, 176))
    for k in range(2):
        cx = W * (0.25 + 0.5 * k)
        d.rectangle([cx - 90, 120, cx + 90, 130], fill=accent)
        fit(d, (cx, 220), name, "sansK", 60, 220, fg)
        text(d, (cx, 290), sub, F("sansB", 24), accent, "mm")
        d.ellipse([cx - 40, 330, cx + 40, 410], fill=accent)
    text(d, (W * 0.5, 470), "185g", F("sansB", 20), fg, "mm")
    return im


def bag_front(name, sub, bg, fg, accent, seed):
    """Front of a snack bag: crimped seals top and bottom, a burst, the snack itself."""
    im = _grad(CELL, CELL, bg, tuple(int(c * 0.8) for c in bg))
    d = ImageDraw.Draw(im)
    r = _rng(seed)
    for y in (0, CELL - 40):
        d.rectangle([0, y, CELL, y + 40], fill=tuple(int(c * 0.7) for c in bg))
        for x in range(0, CELL, 12):
            d.line([x, y, x, y + 40], fill=tuple(int(c * 0.55) for c in bg), width=2)
    # snack pieces
    for _ in range(9):
        cx, cy = r.uniform(110, 400), r.uniform(300, 430)
        d.ellipse([cx - 45, cy - 30, cx + 45, cy + 30], fill=accent, outline=tuple(int(c * 0.8) for c in accent), width=4)
    d.ellipse([360, 70, 490, 200], fill=(255, 230, 40))
    text(d, (425, 120), "NEW", F("sansK", 34), (200, 20, 20), "mm")
    text(d, (425, 160), "期間限定", F("sansB", 18), (200, 20, 20), "mm")
    fit(d, (220, 140), name, "dela", 74, 300, fg)
    d.rounded_rectangle([90, 200, 370, 250], 22, fill=fg)
    text(d, (230, 225), sub, F("sansB", 28), bg, "mm")
    text(d, (60, 470), "60g", F("sansB", 22), fg, "lm")
    return im


def box_front(name, sub, bg, fg, accent, kind, seed):
    im = Image.new("RGB", (CELL, CELL), bg)
    d = ImageDraw.Draw(im)
    if kind == "cookie":
        for k in range(5):
            cx = 90 + k * 85
            d.ellipse([cx - 40, 300, cx + 40, 380], fill=(215, 160, 90), outline=(170, 110, 50), width=4)
            for j in range(4):
                d.ellipse([cx - 20 + j * 10, 320 + (j % 2) * 20, cx - 12 + j * 10, 328 + (j % 2) * 20], fill=(90, 50, 25))
    elif kind == "choc":
        for k in range(4):
            d.rectangle([70 + k * 95, 290, 150 + k * 95, 400], fill=(90, 50, 30), outline=(60, 30, 15), width=4)
    elif kind == "tissue":
        d.ellipse([140, 260, 372, 330], fill=(255, 255, 255))
        for k in range(6):
            d.arc([60 + k * 60, 360, 140 + k * 60, 440], 180, 360, fill=accent, width=6)
    elif kind == "curry":
        d.ellipse([110, 270, 400, 440], fill=(250, 250, 245))
        d.ellipse([150, 300, 300, 410], fill=(245, 240, 230))
        d.chord([230, 290, 380, 420], -60, 120, fill=(170, 100, 30))
    elif kind == "cereal":
        r = _rng(seed)
        d.ellipse([100, 280, 410, 460], fill=(250, 250, 245))
        for _ in range(40):
            cx, cy = r.uniform(140, 370), r.uniform(310, 420)
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(220, 160, 60))
    else:
        d.rectangle([120, 280, 390, 420], fill=accent)
    d.rectangle([0, 0, CELL, 40], fill=accent)
    fit(d, (CELL / 2, 140), name, "maruK", 70, 440, fg)
    text(d, (CELL / 2, 215), sub, F("sansB", 26), fg, "mm")
    text(d, (CELL - 30, 480), "12枚入", F("sansB", 20), fg, "rm")
    return im


def cup_side(name, sub, bg, fg, accent, seed):
    im = Image.new("RGB", (CELL, CELL), bg)
    d = ImageDraw.Draw(im)
    for k in range(2):
        cx = CELL * (0.25 + 0.5 * k)
        d.rectangle([cx - 100, 60, cx + 100, 80], fill=accent)
        fit(d, (cx, 170), name, "dela", 58, 220, fg)
        text(d, (cx, 240), sub, F("sansB", 26), accent, "mm")
        d.ellipse([cx - 70, 300, cx + 70, 440], fill=(240, 220, 170))
        for j in range(5):
            d.arc([cx - 50, 320 + j * 18, cx + 50, 360 + j * 18], 200, 340, fill=(220, 180, 90), width=5)
    return im


def cup_lid(name, bg, fg):
    im = Image.new("RGB", (CELL, CELL), (230, 230, 228))
    d = ImageDraw.Draw(im)
    d.ellipse([16, 16, CELL - 16, CELL - 16], fill=bg)
    d.ellipse([40, 40, CELL - 40, CELL - 40], outline=fg, width=6)
    fit(d, (CELL / 2, CELL / 2 - 20), name, "dela", 64, 380, fg)
    text(d, (CELL / 2, CELL / 2 + 60), "熱湯3分", F("sansB", 30), fg, "mm")
    return im


def onigiri(name, color):
    """The wrapper seen from the front: a triangle of rice in clear film, a seaweed band, label."""
    im = Image.new("RGB", (CELL, CELL), (230, 235, 235))
    d = ImageDraw.Draw(im)
    d.polygon([(256, 40), (480, 470), (32, 470)], fill=(245, 243, 236))
    d.rectangle([150, 300, 362, 470], fill=(30, 40, 30))
    d.rounded_rectangle([140, 200, 372, 280], 14, fill=color)
    text(d, (256, 240), name, F("sansK", 44), (255, 255, 255), "mm")
    d.line([(256, 40), (256, 300)], fill=(200, 40, 40), width=6)
    return im


def sandwich(name, color):
    im = Image.new("RGB", (CELL, CELL), (230, 235, 235))
    d = ImageDraw.Draw(im)
    d.polygon([(30, 470), (482, 470), (256, 60)], fill=(240, 225, 190))
    d.polygon([(60, 440), (452, 440), (256, 110)], fill=(250, 245, 230))
    for k, c in enumerate(((240, 200, 70), (90, 160, 60), (230, 120, 110))):
        y = 300 + k * 40
        d.rectangle([140 - k * 15, y, 372 + k * 15, y + 25], fill=c)
    d.rounded_rectangle([150, 160, 362, 220], 12, fill=color)
    text(d, (256, 190), name, F("sansB", 34), (255, 255, 255), "mm")
    return im


def bento(seed):
    r = _rng(seed)
    im = Image.new("RGB", (CELL, CELL), (40, 40, 42))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 20, CELL - 20, CELL - 20], 30, fill=(30, 30, 32))
    d.rounded_rectangle([40, 40, 250, 470], 20, fill=(245, 243, 236))
    d.ellipse([125, 230, 165, 270], fill=(170, 30, 50))
    items = [(200, 110, 50), (90, 150, 60), (240, 190, 60), (230, 120, 100), (130, 80, 40)]
    for k in range(5):
        x0, y0 = 270, 40 + k * 86
        d.rounded_rectangle([x0, y0, CELL - 40, y0 + 78], 14, fill=items[k])
        for j in range(4):
            cx, cy = r.uniform(x0 + 20, CELL - 60), r.uniform(y0 + 15, y0 + 60)
            d.ellipse([cx - 12, cy - 9, cx + 12, cy + 9], fill=tuple(min(255, c + 30) for c in items[k]))
    d.rounded_rectangle([60, 400, 240, 455], 10, fill=(250, 220, 60))
    text(d, (150, 428), "幕の内 580円", F("sansB", 24), (30, 30, 30), "mm")
    return im


def drug_item(name, sub, bg, fg, accent, kind, seed):
    im = _grad(CELL, CELL, bg, tuple(int(c * 0.9) for c in bg))
    d = ImageDraw.Draw(im)
    if kind == "bottle":
        for k in range(2):
            cx = CELL * (0.25 + 0.5 * k)
            fit(d, (cx, 160), name, "serifB", 50, 220, fg)
            text(d, (cx, 220), sub, F("sansM", 22), fg, "mm")
            d.ellipse([cx - 40, 280, cx + 40, 360], outline=accent, width=6)
            d.ellipse([cx - 18, 302, cx + 18, 338], fill=accent)
    else:
        d.rectangle([0, 0, CELL, 60], fill=accent)
        text(d, (CELL / 2, 30), "第2類医薬品" if kind == "med" else "医薬部外品", F("sansB", 26), (255, 255, 255), "mm")
        fit(d, (CELL / 2, 180), name, "sansK", 66, 440, fg)
        text(d, (CELL / 2, 250), sub, F("sansB", 26), accent, "mm")
        d.rounded_rectangle([140, 300, 372, 420], 20, fill=accent)
        text(d, (CELL / 2, 360), "24錠" if kind == "med" else "30枚", F("sansK", 44), (255, 255, 255), "mm")
    return im


def magazine(title, bg, fg, seed):
    """A magazine cover: a photo of a street at dusk (blocks of buildings, lit windows) under the title."""
    r = _rng(seed)
    im = _grad(CELL, CELL, (40 + seed % 3 * 30, 60, 110), (230, 150, 110))
    d = ImageDraw.Draw(im)
    x = 0
    while x < CELL:
        w = int(r.integers(40, 110))
        h = int(r.integers(120, 330))
        c = tuple(int(v) for v in r.integers(20, 70, 3))
        d.rectangle([x, CELL - h, x + w, CELL], fill=c)
        for wy in range(CELL - h + 12, CELL - 10, 22):
            for wx in range(x + 8, x + w - 10, 16):
                if r.random() < 0.45:
                    d.rectangle([wx, wy, wx + 7, wy + 10], fill=(250, 220, 140))
        x += w + 4
    d.rectangle([0, 0, CELL, 120], fill=bg)
    fit(d, (CELL / 2, 62), title, "dela", 84, CELL - 40, fg)
    for k, t_ in enumerate(["春の新作", "特集 地下街", "保存版"]):
        text(d, (30, 170 + k * 56), t_, F("sansK", 34), (255, 255, 255), "lm")
    text(d, (CELL - 30, 480), "680円", F("sansB", 24), (255, 255, 255), "rm")
    return im


def swatch_cell():
    im = Image.new("RGB", (CELL, CELL), (128, 128, 128))
    d = ImageDraw.Draw(im)
    s = CELL // 8
    for i, (_, c) in enumerate(SWATCHES):
        gx, gy = i % 8, i // 8
        d.rectangle([gx * s, gy * s, gx * s + s - 1, gy * s + s - 1], fill=c)
    return im


def crust(seed, base):
    r = _rng(seed)
    a = np.ones((CELL, CELL, 3), np.float32) * np.array(base, np.float32)
    n = r.normal(0, 1, (CELL // 8, CELL // 8))
    n = np.array(Image.fromarray(((n - n.min()) / (np.ptp(n) + 1e-6) * 255).astype(np.uint8)).resize((CELL, CELL), Image.BICUBIC)) / 255.0
    a *= (0.75 + 0.45 * n)[..., None]
    for _ in range(400):
        x, y = r.integers(0, CELL, 2)
        a[max(0, y - 2):y + 2, max(0, x - 6):x + 6] *= 1.25
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


BOTTLES = [("天然水", "南アルプスの湧き水", (230, 242, 250), (30, 80, 160), (80, 140, 200), "mountain"),
           ("緑茶", "まろやか一番摘み", (40, 120, 60), (255, 255, 255), (150, 200, 90), "leaf"),
           ("烏龍茶", "香ばしい焙煎", (150, 60, 30), (255, 240, 220), (220, 160, 80), "leaf"),
           ("スパークル", "強炭酸 ストロング", (30, 30, 34), (240, 40, 40), (240, 40, 40), "bubbles"),
           ("レモンソーダ", "瀬戸内レモン", (250, 240, 150), (40, 120, 60), (250, 200, 30), "fruit"),
           ("ミルクティー", "ロイヤル", (240, 225, 200), (120, 60, 30), (160, 100, 60), "cup"),
           ("オレンジ100", "果汁100%", (250, 160, 40), (255, 255, 255), (240, 110, 20), "fruit"),
           ("スポーツ飲料", "すばやく水分補給", (230, 240, 250), (20, 70, 160), (60, 150, 220), "bubbles")]
CANS = [("微糖", "炭火焙煎コーヒー", (30, 25, 22), (230, 200, 120), (200, 160, 70)),
        ("ブラック", "深煎り無糖", (15, 15, 15), (255, 255, 255), (180, 180, 180)),
        ("サイダー", "すっきり強炭酸", (220, 235, 245), (20, 80, 170), (40, 120, 220)),
        ("緑茶", "抹茶入り", (40, 110, 60), (255, 255, 255), (170, 210, 100)),
        ("エナジー", "ENERGY", (20, 40, 30), (120, 255, 80), (120, 255, 80)),
        ("カフェオレ", "北海道牛乳", (240, 230, 210), (110, 60, 30), (180, 120, 70)),
        ("ピーチ", "白桃スカッシュ", (250, 220, 220), (220, 80, 110), (240, 150, 160)),
        ("コーラ", "ZERO", (190, 20, 25), (255, 255, 255), (30, 30, 30))]
BAGS = [("ポテチ", "うすしお味", (230, 40, 40), (255, 230, 60), (240, 200, 90)),
        ("ポテチ", "コンソメ", (240, 140, 30), (255, 255, 255), (230, 180, 80)),
        ("のり塩", "瀬戸内産あおさ", (40, 120, 60), (255, 255, 255), (220, 200, 100)),
        ("コーンスナック", "とうもろこし", (250, 210, 40), (200, 40, 40), (250, 180, 40)),
        ("チョコパイ", "濃厚ショコラ", (90, 50, 30), (255, 220, 180), (130, 80, 40)),
        ("せんべい", "しょうゆ味", (240, 230, 210), (130, 60, 30), (190, 130, 60)),
        ("グミ", "フルーツミックス", (240, 120, 170), (255, 255, 255), (250, 80, 120)),
        ("プレッツェル", "バター", (40, 80, 170), (255, 230, 100), (200, 140, 70))]
BOXES = [("クッキー", "バター香る", (250, 240, 220), (140, 80, 40), (200, 60, 50), "cookie"),
         ("チョコ", "ミルクチョコレート", (110, 30, 30), (250, 220, 180), (200, 160, 90), "choc"),
         ("ビスケット", "全粒粉", (60, 110, 170), (255, 255, 255), (250, 200, 60), "cookie"),
         ("ティッシュ", "やわらか 200組", (240, 245, 250), (60, 120, 190), (120, 180, 230), "tissue"),
         ("カレー", "中辛 レトルト", (200, 120, 20), (255, 255, 255), (120, 60, 20), "curry"),
         ("シリアル", "フルーツグラノーラ", (250, 200, 60), (120, 50, 20), (230, 90, 40), "cereal"),
         ("紅茶", "ティーバッグ 25袋", (30, 60, 40), (240, 220, 160), (200, 170, 90), "plain"),
         ("ガム", "ミント", (240, 250, 250), (20, 140, 120), (40, 180, 160), "plain")]
CUPS = [("しょうゆ", "定番の味", (250, 250, 248), (200, 30, 30), (200, 30, 30)),
        ("シーフード", "具だくさん", (230, 240, 250), (20, 70, 150), (40, 110, 200)),
        ("カレー", "スパイシー", (250, 230, 120), (150, 60, 10), (200, 100, 20)),
        ("とんこつ", "濃厚スープ", (30, 30, 30), (250, 210, 60), (250, 210, 60))]
DRUG = [("さらり髪", "シャンプー", (250, 250, 250), (60, 120, 170), (120, 180, 220), "bottle"),
        ("ボディソープ", "ナチュラル", (230, 245, 230), (60, 120, 70), (120, 190, 120), "bottle"),
        ("かぜぐすり", "総合かぜ薬", (240, 250, 255), (20, 90, 180), (230, 60, 50), "med"),
        ("胃腸薬", "すっきり", (250, 250, 240), (30, 120, 60), (40, 150, 80), "med"),
        ("目ぐすり", "クール", (230, 240, 255), (30, 60, 160), (60, 160, 230), "med"),
        ("歯みがき", "ホワイトニング", (250, 250, 255), (220, 40, 60), (40, 120, 220), "other"),
        ("マスク", "ふつうサイズ", (240, 245, 250), (60, 110, 170), (100, 160, 220), "other"),
        ("化粧水", "しっとり", (250, 240, 245), (170, 90, 120), (220, 160, 190), "bottle")]
MAGS = [("Hikari", (240, 80, 110), (255, 255, 255)), ("SPOT", (30, 30, 30), (250, 220, 60)),
        ("月刊ちかがい", (60, 130, 190), (255, 255, 255)), ("CAR LIFE", (200, 30, 30), (255, 255, 255))]


def build_goods():
    os.makedirs(OUT, exist_ok=True)
    A = Image.new("RGB", (CELL * N, CELL * N), (128, 128, 128))

    def put(im, col, row):
        A.paste(im, (col * CELL, row * CELL))
    for i, b in enumerate(BOTTLES):
        put(bottle_label(*b, seed=10 + i), i, 0)
    for i, c in enumerate(CANS):
        put(can_label(*c, seed=20 + i), i, 1)
    for i, b in enumerate(BAGS):
        put(bag_front(*b, seed=30 + i), i, 2)
    for i, b in enumerate(BOXES):
        put(box_front(*b, seed=40 + i), i, 3)
    for i, c in enumerate(CUPS):
        put(cup_side(*c, seed=50 + i), i, 4)
        put(cup_lid(c[0], c[2], c[3]), 4 + i, 4)
    for i, (n_, col) in enumerate((("鮭", (220, 90, 60)), ("梅", (170, 30, 60)), ("ツナマヨ", (40, 100, 170)), ("昆布", (60, 80, 40)))):
        put(onigiri(n_, col), i, 5)
    put(sandwich("ミックス", (40, 120, 60)), 4, 5)
    put(sandwich("たまご", (230, 170, 30)), 5, 5)
    put(bento(61), 6, 5)
    put(bento(62), 7, 5)
    for i, g in enumerate(DRUG):
        put(drug_item(*g, seed=70 + i), i, 6)
    for i, mg in enumerate(MAGS):
        put(magazine(*mg, seed=80 + i), i, 7)
    put(swatch_cell(), *SWATCH_CELL)
    put(crust(5, (205, 150, 80)), 5, 7)
    put(crust(6, (160, 95, 45)), 6, 7)
    put(crust(7, (235, 200, 140)), 7, 7)
    p = os.path.join(OUT, "goods.png")
    A.save(p)
    return p


def build_books():
    """64 spines (title, author, publisher mark) and 8 covers."""
    os.makedirs(OUT, exist_ok=True)
    W = 2048
    A = Image.new("RGB", (W, W), (200, 195, 185))
    d = ImageDraw.Draw(A)
    r = _rng(3)
    titles = ["地下の街", "灯りの記憶", "迷宮案内", "夜の駅", "古い地図", "白い通路", "七番目の扉", "影の商店街",
              "猫と改札", "終電のあと", "東の出口", "雨の日の本屋", "ひかりの広場", "名もなき階段", "遠い汽笛", "春の地下道"]
    pal = [(150, 30, 30), (30, 60, 110), (20, 90, 60), (230, 210, 170), (40, 40, 45), (200, 150, 40), (110, 60, 120),
           (240, 240, 235), (80, 110, 140), (170, 80, 40)]
    for i in range(64):
        col, row = i % 32, i // 32
        x0, y0 = col * 64, row * 512
        bg = pal[int(r.integers(len(pal)))]
        fg = (255, 255, 255) if sum(bg) < 360 else (30, 30, 30)
        d.rectangle([x0, y0, x0 + 63, y0 + 511], fill=bg)
        d.rectangle([x0, y0 + 30, x0 + 63, y0 + 36], fill=fg)
        t = titles[i % len(titles)]
        f = F("serifB" if i % 3 else "sansB", 30)
        for k, ch in enumerate(t[:8]):
            text(d, (x0 + 32, y0 + 70 + k * 36), ch, f, fg, "mm")
        for k, ch in enumerate(["著", "者", "名"][:2 + i % 2]):
            text(d, (x0 + 32, y0 + 390 + k * 24), ch, F("sansM", 18), fg, "mm")
        d.rectangle([x0 + 18, y0 + 470, x0 + 46, y0 + 498], outline=fg, width=3)
    for i in range(8):
        x0, y0 = i * 256, 1024
        bg = pal[(i * 3) % len(pal)]
        fg = (255, 255, 255) if sum(bg) < 360 else (30, 30, 30)
        cover = Image.new("RGB", (256, 512), bg)
        dc = ImageDraw.Draw(cover)
        dc.ellipse([40, 200, 216, 376], fill=tuple(min(255, c + 60) for c in bg))
        text(dc, (128, 80), titles[(i * 5) % 16], F("serifB", 34 if len(titles[(i * 5) % 16]) <= 5 else 26), fg, "mm")
        text(dc, (128, 440), "著者名", F("sansM", 22), fg, "mm")
        A.paste(cover, (x0, y0))
    p = os.path.join(OUT, "books.png")
    # kept at 3072 so the app export keeps its detail (textures under 3000 px go down to 1024)
    A.resize((3072, 3072), Image.LANCZOS).save(p)
    return p


def spine_uv(i):
    """UV box (u0, v0, u1, v1) of spine i (0..63), Blender v up."""
    col, row = i % 32, i // 32
    W = 2048.0
    u0, u1 = (col * 64 + 1) / W, (col * 64 + 63) / W
    v1 = 1.0 - (row * 512 + 1) / W
    v0 = 1.0 - (row * 512 + 511) / W
    return (u0, v0, u1, v1)


def cover_uv(i):
    W = 2048.0
    u0, u1 = (i * 256 + 1) / W, (i * 256 + 255) / W
    v1 = 1.0 - (1024 + 1) / W
    v0 = 1.0 - (1024 + 511) / W
    return (u0, v0, u1, v1)


if __name__ == "__main__":
    print(build_goods(), build_books())
