"""Signs, shop fronts and shelf faces for the underground maze (drawn with Pillow).

Everything here is made up: the operator is the fictional 陽ヶ丘地下鉄 (Higaoka Metro, see
scripts/graphics.py), the mall is ひかり地下街, and the shops are invented names (no real chain's
name, colours or logo).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from graphics import F, text, arrow, grad, add_print_texture  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "cache", "dungeon", "gfx")
LINE = (0, 150, 110)        # 陽ヶ丘線 line colour (green)

# id, type, name, sub, fascia background, text colour, accent
SHOPS = [
    ("mart", "konbini", "まちかどマート", "MACHIKADO MART  24h", (250, 250, 248), (20, 120, 70), (240, 120, 30)),
    ("cafe", "cafe", "珈琲 あかり", "AKARI COFFEE", (52, 36, 28), (240, 225, 195), (200, 150, 90)),
    ("bake", "bakery", "パン工房 こむぎ", "Boulangerie KOMUGI", (246, 236, 214), (110, 70, 35), (200, 140, 60)),
    ("drug", "drug", "くすりのハル", "HARU DRUG", (20, 80, 170), (255, 255, 255), (255, 210, 0)),
    ("book", "books", "ひらく書店", "HIRAKU BOOKS", (28, 70, 52), (245, 245, 235), (210, 190, 120)),
    ("ramen", "ramen", "らーめん 一筋", "RAMEN HITOSUJI", (18, 18, 18), (245, 240, 230), (200, 30, 30)),
    ("wear", "clothes", "ATELIER NOVA", "women & men", (236, 236, 234), (30, 30, 30), (150, 150, 150)),
    ("soba", "soba", "立ちそば ほたる", "HOTARU SOBA", (230, 222, 200), (40, 40, 40), (30, 90, 60)),
    ("flower", "flowers", "花のみどり", "FLOWERS MIDORI", (250, 250, 245), (60, 110, 60), (230, 120, 150)),
    ("shut", "shutter", "", "", (0, 0, 0), (0, 0, 0), (0, 0, 0)),
]
SHOP = {s[0]: s for s in SHOPS}


def fascia(sid):
    """The lit sign band over a shop front (8 m x 0.5 m)."""
    _, kind, name, sub, bg, fg, ac = SHOP[sid]
    W, H = 3200, 200
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if kind == "konbini":
        d.rectangle([0, H - 34, W, H - 20], fill=ac)
        d.rectangle([0, H - 20, W, H], fill=fg)
    if kind == "ramen":
        d.rectangle([0, 0, W, 14], fill=ac)
        d.rectangle([0, H - 14, W, H], fill=ac)
    font = F("serifB" if kind in ("cafe", "books", "soba") else ("maruK" if kind in ("bakery", "flowers") else "sansK"), 128)
    if kind == "clothes":
        font = F("sansL", 120)
    text(d, (W / 2 - 160 if sub else W / 2, H / 2), name, font, fg, "mm", spacing=8 if kind == "clothes" else 0)
    tw = d.textlength(name, font=font)
    if sub:
        text(d, (W / 2 - 160 + tw / 2 + 70, H / 2 + 8), sub, F("sansM", 46), ac if kind != "konbini" else fg, "lm")
    return im


def shop_sign_board(sid):
    """Small square sign hanging out from the front (seen along the street), 0.6 x 0.6 m."""
    _, kind, name, sub, bg, fg, ac = SHOP[sid]
    W = H = 600
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, W - 12, H - 12], outline=ac, width=10)
    words = name.split(" ") if " " in name else [name[:len(name) // 2 + len(name) % 2], name[len(name) // 2 + len(name) % 2:]]
    for i, w in enumerate(words[:2]):
        text(d, (W / 2, 200 + i * 170), w, F("sansK", 130 if len(w) <= 4 else 96), fg, "mm")
    return im


def shelf_face(kind, seed, W=2048, H=512):
    """Front of one shelf row full of goods (1.2 m x 0.3 m): packs, bottles, books, bread..."""
    rng = np.random.default_rng(seed)
    bgc = {"books": (60, 50, 40), "bakery": (120, 85, 50), "drinks": (20, 22, 26)}.get(kind, (235, 235, 232))
    im = Image.new("RGB", (W, H), bgc)
    d = ImageDraw.Draw(im)
    x = 6
    pal = [(220, 40, 40), (250, 200, 40), (40, 120, 220), (60, 170, 80), (240, 240, 240), (250, 130, 30),
           (140, 60, 160), (30, 30, 30), (230, 90, 140), (120, 200, 230), (190, 160, 110)]
    while x < W - 10:
        if kind == "books":
            w = int(rng.integers(26, 70))
            h = int(rng.integers(300, 490))
            c = pal[int(rng.integers(len(pal)))]
            c = tuple(int(v * rng.uniform(0.55, 0.95)) for v in c)
            d.rectangle([x, H - h, x + w, H - 4], fill=c)
            for k in range(int(rng.integers(1, 4))):
                yy = H - h + int(rng.integers(20, max(21, h - 60)))
                d.rectangle([x + 5, yy, x + w - 5, yy + 8], fill=(230, 220, 190))
            x += w + 2
        elif kind == "bakery":
            w = int(rng.integers(140, 220))
            h = int(rng.integers(90, 170))
            c = (int(rng.integers(150, 205)), int(rng.integers(90, 130)), int(rng.integers(30, 60)))
            d.rounded_rectangle([x, H - 40 - h, x + w, H - 40], int(h * 0.45), fill=c)
            d.ellipse([x + w * 0.2, H - 40 - h * 0.85, x + w * 0.8, H - 40 - h * 0.45], fill=tuple(min(255, v + 40) for v in c))
            d.rectangle([x - 6, H - 40, x + w + 6, H - 28], fill=(200, 200, 195))
            x += w + 18
        elif kind == "drinks":
            w = int(rng.integers(60, 96))
            h = int(rng.integers(300, 440))
            c = pal[int(rng.integers(len(pal)))]
            d.rounded_rectangle([x + 8, H - h, x + w - 8, H - h + 60], 12, fill=(240, 240, 240))
            d.rounded_rectangle([x, H - h + 50, x + w, H - 8], 18, fill=c)
            d.rectangle([x + 6, H - h * 0.6, x + w - 6, H - h * 0.35], fill=(250, 250, 250))
            x += w + 10
        else:      # packs and boxes (konbini, drug store)
            w = int(rng.integers(90, 180))
            h = int(rng.integers(150, 420))
            c = pal[int(rng.integers(len(pal)))]
            d.rectangle([x, H - h, x + w, H - 8], fill=c)
            c2 = pal[int(rng.integers(len(pal)))]
            d.rectangle([x + 10, H - h + 20, x + w - 10, H - h + 20 + h * 0.3], fill=c2)
            d.ellipse([x + w * 0.25, H - h * 0.5, x + w * 0.75, H - h * 0.5 + w * 0.5], fill=(250, 250, 250))
            x += w + int(rng.integers(4, 14))
    # price rail
    if kind not in ("books",):
        d.rectangle([0, H - 30, W, H], fill=(250, 250, 250))
        for k in range(0, W, 170):
            d.rectangle([k + 20, H - 26, k + 120, H - 6], fill=(255, 230, 60))
    return add_print_texture(im.filter(ImageFilter.GaussianBlur(0.6)), seed, 2.5)


def menu_board(sid):
    """Menu over the counter (café chalkboard, ramen / soba wooden tags), 2.4 x 0.6 m."""
    _, kind, name, sub, bg, fg, ac = SHOP[sid]
    W, H = 2400, 600
    if kind == "cafe":
        im = Image.new("RGB", (W, H), (32, 36, 34))
        d = ImageDraw.Draw(im)
        items = [("ブレンド", "420"), ("カフェラテ", "480"), ("カプチーノ", "480"), ("アイスコーヒー", "450"),
                 ("本日のケーキ", "520"), ("ホットサンド", "580")]
        for i, (a, p) in enumerate(items):
            cx, cy = 60 + (i % 2) * 1180, 80 + (i // 2) * 170
            text(d, (cx, cy), a, F("maruM", 80), (235, 235, 225))
            text(d, (cx + 1050, cy), p, F("maruM", 80), (240, 200, 120), "ra")
        return add_print_texture(im, 31, 4)
    im = Image.new("RGB", (W, H), (200, 170, 120) if kind == "soba" else (230, 205, 150))
    d = ImageDraw.Draw(im)
    items = (["かけそば 380", "きつね 450", "たぬき 450", "天ぷら 520", "月見 450", "カレー丼 480"] if kind == "soba"
             else ["醤油らーめん 850", "塩らーめん 850", "味噌らーめん 900", "つけ麺 950", "餃子 350", "ライス 150"])
    for i, s in enumerate(items):
        x0 = 40 + i * 393
        d.rectangle([x0, 20, x0 + 360, H - 20], fill=(245, 235, 210), outline=(120, 80, 40), width=6)
        a, p = s.rsplit(" ", 1)
        for j, ch in enumerate(a[:6]):
            text(d, (x0 + 180, 70 + j * 72), ch, F("serifB", 64), (30, 20, 10), "mm")
        text(d, (x0 + 180, H - 60), p + "円", F("sansB", 48), (180, 30, 20), "mm")
    return add_print_texture(im, 32, 3)


def noren(sid):
    _, kind, name, sub, bg, fg, ac = SHOP[sid]
    W, H = 1600, 700
    im = Image.new("RGB", (W, H), (25, 30, 60) if kind == "soba" else (20, 20, 20))
    d = ImageDraw.Draw(im)
    for k in range(1, 4):
        d.rectangle([k * W // 4 - 6, 120, k * W // 4 + 6, H], fill=(10, 10, 10))
    short = name.split(" ")[-1]
    for i, ch in enumerate(short[:4]):
        text(d, (W * (i + 0.5) / max(1, len(short[:4])), H / 2 + 40), ch, F("serifK", 260), (245, 240, 230), "mm")
    return add_print_texture(im, 33, 5)


def shutter_notice():
    W, H = 600, 850
    im = Image.new("RGB", (W, H), (250, 250, 246))
    d = ImageDraw.Draw(im)
    text(d, (W / 2, 120), "テナント募集", F("sansB", 76), (200, 30, 30), "mm")
    for i, s in enumerate(["この区画の出店者を", "募集しています。", "", "ひかり地下街 管理事務所", "TEL 000-000-0000"]):
        text(d, (W / 2, 280 + i * 90), s, F("sansM", 40), (40, 40, 40), "mm")
    return add_print_texture(im, 34, 3)


def station_board(name, kana, en, left, right):
    """Station name board on the track walls (3.0 x 0.7 m)."""
    W, H = 3000, 700
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 150, W, H], fill=LINE)
    d.rounded_rectangle([780, 120, 1080, 420], 60, fill=LINE)
    d.rounded_rectangle([815, 155, 1045, 385], 45, fill=(255, 255, 255))
    text(d, (930, 215), "H", F("sansB", 100), LINE, "mm")
    text(d, (930, 320), "07", F("sansB", 100), LINE, "mm")
    text(d, (W / 2 + 160, 215), name, F("sansB", 280), (20, 20, 20), "mm")
    text(d, (W / 2 + 160, 440), f"{kana}   {en}", F("sansM", 88), (40, 40, 40), "mm")
    text(d, (60, H - 75), f"◀ {left}", F("sansB", 70), (255, 255, 255), "lm")
    text(d, (W - 60, H - 75), f"{right} ▶", F("sansB", 70), (255, 255, 255), "rm")
    return im


def gate_fascia(name, en):
    """Band over the gate line: '陽ヶ丘地下鉄  ○○駅 改札'."""
    W, H = 4800, 400
    im = Image.new("RGB", (W, H), (24, 26, 30))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 70, 330, 330], 40, fill=LINE)
    text(d, (195, 200), "H", F("sansB", 190), (255, 255, 255), "mm")
    text(d, (400, 150), "陽ヶ丘地下鉄", F("sansM", 90), (230, 230, 230), "lm")
    text(d, (400, 285), "Higaoka Metro", F("sansM", 64), (180, 180, 180), "lm")
    text(d, (W / 2 + 300, 170), f"{name}駅  改札", F("sansB", 150), (255, 255, 255), "mm")
    text(d, (W / 2 + 300, 315), f"{en} Station  Ticket Gates", F("sansM", 72), (200, 200, 200), "mm")
    return im


def mall_sign(direction="up"):
    """Hanging direction sign of the mall: ひかり地下街 / 改札 / 出口 (2.4 x 0.4 m)."""
    W, H = 3000, 500
    im = Image.new("RGB", (W, H), (34, 36, 40))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1000, H], fill=(255, 205, 0))
    arrow(d, 170, 250, 280, (20, 20, 20), direction)
    text(d, (360, 200), "出口", F("sansB", 170), (20, 20, 20), "lm")
    text(d, (365, 380), "Exit", F("sansM", 80), (20, 20, 20), "lm")
    d.rounded_rectangle([1080, 110, 1360, 390], 40, fill=LINE)
    text(d, (1220, 250), "H", F("sansB", 200), (255, 255, 255), "mm")
    text(d, (1420, 190), "陽ヶ丘線  改札", F("sansB", 140), (255, 255, 255), "lm")
    text(d, (1425, 370), "Higaoka Line  Gates", F("sansM", 80), (210, 210, 210), "lm")
    return im


def mall_banner():
    W, H = 2400, 600
    im = grad(W, H, (250, 246, 236), (240, 228, 205))
    d = ImageDraw.Draw(im)
    d.ellipse([90, 90, 510, 510], fill=(250, 190, 60))
    d.ellipse([170, 170, 430, 430], fill=(255, 235, 170))
    text(d, (620, 220), "ひかり地下街", F("maruK", 190), (90, 60, 30), "lm")
    text(d, (630, 450), "HIKARI UNDERGROUND MALL", F("sansM", 74), (140, 100, 60), "lm")
    return add_print_texture(im, 41, 2)


def build_all():
    os.makedirs(OUT, exist_ok=True)
    items = {f"fascia_{s[0]}": fascia(s[0]) for s in SHOPS if s[1] != "shutter"}
    items.update({f"board_{s[0]}": shop_sign_board(s[0]) for s in SHOPS if s[1] != "shutter"})
    for k, (kind, seed) in {"shelf_mart": ("packs", 1), "shelf_mart2": ("packs", 2), "shelf_drinks": ("drinks", 3),
                            "shelf_drug": ("packs", 4), "shelf_books": ("books", 5), "shelf_books2": ("books", 6),
                            "shelf_bread": ("bakery", 7)}.items():
        items[k] = shelf_face(kind, seed)
    items["menu_cafe"] = menu_board("cafe")
    items["menu_ramen"] = menu_board("ramen")
    items["menu_soba"] = menu_board("soba")
    items["noren_ramen"] = noren("ramen")
    items["noren_soba"] = noren("soba")
    items["shutter_notice"] = shutter_notice()
    items["station_a"] = station_board("あかり町", "あかりまち", "Akarimachi", "H06 ほたる橋", "H08 しずく台")
    items["gate_a"] = gate_fascia("あかり町", "Akarimachi")
    items["mall_sign"] = mall_sign("up")
    items["mall_banner"] = mall_banner()
    paths = {}
    for k, im in items.items():
        p = os.path.join(OUT, k + ".png")
        im.save(p)
        paths[k] = p
    return paths


if __name__ == "__main__":
    for k, v in build_all().items():
        print(k, v)
