"""Signs and screen content for the Umeda Station reproduction (drawn with Pillow).

Station name boards and platform signs carry factual wayfinding information
(station name, number M16, neighbouring stations, directions). Advertising and
LED-screen content is generic, made up for this render.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from graphics import F, text, arrow, grad, add_print_texture  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache", "umeda", "gfx")
RED = (229, 23, 31)   # Midosuji line colour


def line_badge(d, cx, cy, r, num):
    d.rounded_rectangle([cx - r, cy - r, cx + r, cy + r], int(r * 0.35), fill=RED)
    d.rounded_rectangle([cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78], int(r * 0.25), fill=(255, 255, 255))
    text(d, (cx, cy - r * 0.28), "M", F("sansB", int(r * 0.62)), RED, "mm")
    text(d, (cx, cy + r * 0.32), num, F("sansB", int(r * 0.62)), RED, "mm")


def name_board():
    W, H = 3000, 700     # 3.0 x 0.7 m
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 150, W, H], fill=RED)
    line_badge(d, 1060, 270, 150, "16")
    text(d, (W / 2 + 120, 210), "梅田", F("sansB", 300), (20, 20, 20), "mm")
    text(d, (W / 2 + 120, 430), "うめだ   Umeda", F("sansM", 92), (40, 40, 40), "mm")
    text(d, (60, H - 75), "◀ M15 中津 Nakatsu", F("sansB", 70), (255, 255, 255), "lm")
    text(d, (W - 60, H - 75), "淀屋橋 Yodoyabashi M17 ▶", F("sansB", 70), (255, 255, 255), "rm")
    return im


def track_sign(num, dests, dests_en):
    W, H = 2600, 520
    im = Image.new("RGB", (W, H), (28, 30, 34))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 460, 460], fill=RED)
    text(d, (260, 262), str(num), F("sansB", 300), (255, 255, 255), "mm")
    text(d, (560, 200), dests, F("sansB", 118), (255, 255, 255), "lm")
    text(d, (565, 390), dests_en, F("sansM", 80), (225, 225, 225), "lm")
    return im


def exit_sign():
    W, H = 2400, 480
    im = Image.new("RGB", (W, H), (255, 205, 0))
    d = ImageDraw.Draw(im)
    k = (20, 20, 20)
    arrow(d, 200, 240, 280, k, "up")
    text(d, (390, 210), "出口", F("sansB", 200), k, "lm")
    text(d, (400, 400), "Exit", F("sansM", 80), k, "lm")
    text(d, (880, 200), "中改札・南改札", F("sansB", 130), k, "lm")
    text(d, (885, 380), "Central / South Gates", F("sansM", 80), k, "lm")
    return im


def gate_sign(jp, en):
    W, H = 2600, 480
    im = Image.new("RGB", (W, H), (30, 32, 36))
    d = ImageDraw.Draw(im)
    line_badge(d, 250, 240, 150, "16")
    text(d, (470, 190), jp, F("sansB", 170), (255, 255, 255), "lm")
    text(d, (480, 375), en, F("sansM", 84), (220, 220, 220), "lm")
    return im


def exit_sign_b1(nums):
    W, H = 2400, 480
    im = Image.new("RGB", (W, H), (255, 205, 0))
    d = ImageDraw.Draw(im)
    k = (20, 20, 20)
    arrow(d, 200, 240, 280, k, "up")
    text(d, (390, 210), "出口", F("sansB", 200), k, "lm")
    text(d, (400, 400), "Exit", F("sansM", 80), k, "lm")
    text(d, (900, 230), nums, F("sansB", 210), k, "lm")
    return im


def platform_sign():
    W, H = 2400, 480
    im = Image.new("RGB", (W, H), (30, 32, 36))
    d = ImageDraw.Draw(im)
    line_badge(d, 250, 240, 150, "")
    text(d, (470, 200), "御堂筋線 のりば", F("sansB", 170), (255, 255, 255), "lm")
    text(d, (480, 380), "Midosuji Line Platforms", F("sansM", 84), (220, 220, 220), "lm")
    arrow(d, 2220, 240, 300, (255, 255, 255), "right")
    return im


def led_vision():
    """Wide LED screen content (40 m x 4 m): a generic night-city panorama."""
    W, H = 4000, 400
    rng = np.random.default_rng(3)
    y = np.linspace(0, 1, H)[:, None, None]
    sky = (np.array([8, 14, 40]) * (1 - y) + np.array([60, 40, 110]) * y).repeat(W, 1)
    a = sky.astype(np.float32)
    # glowing horizon
    a += (np.exp(-((y[..., 0] - 0.78) / 0.12) ** 2)[..., None] * np.array([120, 60, 90])).repeat(W, 1)
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    x = 0
    while x < W:
        bw = int(rng.uniform(40, 140)); bh = int(rng.uniform(80, 300))
        col = tuple(int(c) for c in rng.uniform(10, 35, 3))
        d.rectangle([x, H - bh, x + bw, H], fill=col)
        for wy in range(H - bh + 8, H - 6, 12):
            for wx in range(x + 6, x + bw - 6, 10):
                if rng.random() < 0.45:
                    c = (255, 225, 150) if rng.random() < 0.8 else (170, 220, 255)
                    d.rectangle([wx, wy, wx + 4, wy + 5], fill=c)
        x += bw + int(rng.uniform(2, 12))
    for _ in range(140):
        sx, sy = rng.uniform(0, W), rng.uniform(5, 120)
        d.point((sx, sy), fill=(255, 255, 255))
    glow = img.filter(ImageFilter.GaussianBlur(6))
    img = Image.blend(img, glow, 0.25)
    d = ImageDraw.Draw(img)
    text(d, (W * 0.5, 90), "OSAKA  NIGHT  VIEW", F("sansL", 90), (255, 255, 255), "mm", spacing=30)
    return img


def ad_panel(kind):
    W, H = 1800, 1000
    if kind == 0:
        im = grad(W, H, (250, 120, 60), (250, 210, 120))
        d = ImageDraw.Draw(im)
        d.ellipse([1150, 200, 1650, 700], fill=(255, 240, 200))
        text(d, (90, 200), "夏の果実、", F("maruK", 120), (255, 255, 255))
        text(d, (90, 360), "はじけるソーダ。", F("maruK", 120), (255, 255, 255))
        d.rounded_rectangle([1290, 300, 1500, 760], 40, fill=(255, 255, 255))
        d.rounded_rectangle([1310, 360, 1480, 740], 30, fill=(250, 150, 60))
        text(d, (90, 820), "期間限定  全国のコンビニで", F("sansB", 64), (120, 40, 10))
    elif kind == 1:
        im = grad(W, H, (20, 40, 70), (40, 90, 140))
        d = ImageDraw.Draw(im)
        for i in range(9):
            x = 900 + i * 90
            d.rectangle([x, 900 - (i * 53) % 400 - 250, x + 70, 900], fill=(15, 25, 45))
        text(d, (90, 220), "はじめての", F("sansB", 120), (255, 255, 255))
        text(d, (90, 380), "英会話、駅前で。", F("sansB", 120), (255, 255, 255))
        d.rounded_rectangle([90, 620, 760, 740], 20, fill=(255, 205, 0))
        text(d, (425, 680), "無料体験レッスン受付中", F("sansB", 56), (20, 30, 50), "mm")
    else:
        im = Image.new("RGB", (W, H), (245, 243, 238))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 720, H], fill=(40, 70, 60))
        for k in range(6):
            r = 260 - k * 40
            d.ellipse([360 - r, 500 - r, 360 + r, 500 + r], outline=(200, 220, 200), width=6)
        text(d, (820, 260), "暮らしに、", F("serifB", 120), (40, 60, 50))
        text(d, (820, 420), "緑をひとつ。", F("serifB", 120), (40, 60, 50))
        text(d, (825, 640), "インテリアグリーンフェア 開催中", F("sansM", 54), (80, 90, 80))
    return add_print_texture(im, 10 + kind, 2.0)


def build_all():
    os.makedirs(OUT, exist_ok=True)
    items = {
        "name_board": name_board(),
        "track1": track_sign(1, "なんば・天王寺・なかもず方面", "for Namba, Tennoji & Nakamozu"),
        "track2": track_sign(2, "新大阪・千里中央・箕面萱野方面", "for Shin-Osaka & Minoh-kayano"),
        "exit_sign": exit_sign(),
        "gate_nw": gate_sign("中北西改札", "Central North-West Gate"),
        "gate_ne": gate_sign("中北東改札", "Central North-East Gate"),
        "gate_n": gate_sign("北改札", "North Gate"),
        "gate_s": gate_sign("南改札", "South Gate"),
        "exit_b1": exit_sign_b1("6・7・8・9"),
        "plat_sign": platform_sign(),
        "led_vision": led_vision(),
        "ad0": ad_panel(0), "ad1": ad_panel(1), "ad2": ad_panel(2),
    }
    paths = {}
    for k, im in items.items():
        p = os.path.join(OUT, k + ".png")
        im.save(p)
        paths[k] = p
    return paths


if __name__ == "__main__":
    for k, v in build_all().items():
        print(k, v)
