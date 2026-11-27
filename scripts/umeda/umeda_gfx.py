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


def aerial_city(W, H, seed):
    """Dusk aerial view of a city, painted procedurally (dense blocks, streets, lit windows)."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, H)[:, None, None]
    img = (np.array([40, 52, 86]) * (1 - y) + np.array([92, 86, 110]) * y).repeat(W, 1).astype(np.float32)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(int(W * H / 260)):
        x0, y0 = rng.uniform(0, W), rng.uniform(H * 0.18, H)
        s = (0.3 + y0 / H) * rng.uniform(10, 34)
        g = rng.uniform(40, 120)
        col = (int(g * rng.uniform(0.8, 1.0)), int(g * rng.uniform(0.85, 1.0)), int(g * rng.uniform(0.95, 1.2)))
        d.rectangle([x0, y0, x0 + s * rng.uniform(0.8, 2.2), y0 + s * rng.uniform(0.6, 1.4)], fill=col)
    for _ in range(int(W * H / 90)):
        x0, y0 = rng.uniform(0, W), rng.uniform(H * 0.2, H)
        c = (255, int(rng.uniform(190, 235)), int(rng.uniform(120, 190))) if rng.random() < 0.75 else (190, 220, 255)
        d.point((x0, y0), fill=c)
    for k in range(6):            # avenues of light
        x0 = rng.uniform(0, W)
        d.line([(x0, H), (x0 + rng.uniform(-200, 200), H * 0.2)], fill=(255, 205, 140), width=3)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    sky = Image.new("RGB", (W, int(H * 0.22)))
    sd = ImageDraw.Draw(sky)
    for yy in range(sky.height):
        t = yy / sky.height
        sd.line([(0, yy), (W, yy)], fill=(int(250 - 90 * t), int(170 - 80 * t), int(150 - 40 * t)))
    im.paste(sky, (0, 0))
    return im


def led_campaign():
    """Content for the 40 m x 4 m LED screen: a fictional airline/travel campaign in flat graphics."""
    W, H = 6000, 600
    im = Image.new("RGB", (W, H), (248, 248, 250))
    left = grad(1800, H, (210, 30, 125), (120, 40, 175))
    im.paste(left, (0, 0))
    mid = grad(2300, H, (255, 150, 90), (120, 70, 170))
    im.paste(mid, (1800, 0))
    d = ImageDraw.Draw(im)
    # sunset over a stylised skyline in the middle panel
    d.ellipse([2700, 160, 3100, 560], fill=(255, 225, 150))
    rng = np.random.default_rng(6)
    x = 1800
    while x < 4100:
        bw, bh = int(rng.uniform(60, 160)), int(rng.uniform(90, 330))
        d.rectangle([x, H - bh, x + bw, H], fill=(70, 45, 95))
        x += bw + int(rng.uniform(0, 18))
    d.polygon([(1800, H), (4100, H), (4100, 520), (1800, 560)], fill=(55, 35, 80))
    text(d, (140, 150), "旅は、", F("sansB", 170), (255, 255, 255))
    text(d, (140, 340), "この駅から。", F("sansB", 170), (255, 255, 255))
    d.rectangle([4100, 0, W, H], fill=(248, 248, 250))
    text(d, (5050, 230), "SKY OSAKA", F("sansB", 200), (30, 40, 90), "mm")
    text(d, (5050, 420), "関西から、世界の街へ。", F("sansM", 80), (60, 60, 80), "mm")
    d.ellipse([4230, 360, 4350, 480], fill=(40, 110, 220))
    return im


def band_ad(kind):
    """Backlit ads along the lower arch above the track wall (7.6 m x 1.6 m)."""
    W, H = 2850, 600
    if kind == 0:
        im = Image.new("RGB", (W, H), (240, 238, 232))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 1300, H], fill=(170, 190, 205))
        for i in range(7):
            x = 120 + i * 150
            h = 260 + (i * 97) % 220
            d.rectangle([x, H - h, x + 110, H], fill=(70 + i * 8, 85 + i * 6, 105 + i * 5))
            for wy in range(H - h + 12, H - 8, 22):
                d.rectangle([x + 10, wy, x + 100, wy + 8], fill=(200, 215, 225))
        text(d, (1400, 200), "駅直結・直上1分", F("serifB", 150), (40, 40, 45))
        text(d, (1405, 420), "梅田の新しい住まい  モデルルーム公開中", F("sansM", 70), (60, 60, 65))
    elif kind == 1:
        im = grad(W, H, (20, 90, 160), (60, 170, 210))
        d = ImageDraw.Draw(im)
        for k in range(5):
            d.ellipse([2100 + k * 60, 80 + k * 20, 2600 - k * 40, 560 - k * 10], outline=(255, 255, 255), width=8)
        text(d, (120, 190), "この夏、海へ。", F("sansB", 170), (255, 255, 255))
        text(d, (125, 420), "直行バス 毎日運行  予約受付中", F("sansM", 72), (230, 245, 255))
    else:
        im = Image.new("RGB", (W, H), (250, 225, 60))
        d = ImageDraw.Draw(im)
        d.rectangle([1900, 0, W, H], fill=(230, 60, 50))
        text(d, (120, 200), "スポーツ・リハビリ・介護", F("sansB", 140), (30, 30, 30))
        text(d, (125, 420), "専門学校 オープンキャンパス開催", F("sansM", 72), (40, 40, 40))
        text(d, (2370, 300), "見学OK", F("sansB", 150), (255, 255, 255), "mm")
    return add_print_texture(im, 30 + kind, 2.0)


def enrich(im, seed):
    """Fine print, a QR code, a small logo and a legal line: the dense small information that
    real posters carry and that shows when the image is enlarged."""
    rng = np.random.default_rng(seed)
    W, H = im.size
    d = ImageDraw.Draw(im)
    fh = max(12, H // 60)
    y0 = H - int(H * 0.085)
    d.rectangle([0, y0, W, H], fill=(250, 250, 250))
    words = ["詳しくはウェブサイトをご覧ください", "※画像はイメージです", "価格は税込表示", "一部店舗を除く",
             "なくなり次第終了", "お問い合わせ 0120-000-000", "受付時間 10:00-18:00", "© 2024"]
    x = int(W * 0.03)
    yy = y0 + int(fh * 0.6)
    for k in range(3):
        line = "  ".join(rng.choice(words, 3, replace=False))
        text(d, (x, yy + k * int(fh * 1.35)), line, F("sansM", fh), (80, 80, 80))
    q = int(H * 0.075)
    qx, qy = W - q - int(W * 0.02), y0 + (H - y0 - q) // 2
    d.rectangle([qx, qy, qx + q, qy + q], fill=(255, 255, 255))
    cells = 21
    cs = q / cells
    bits = rng.random((cells, cells)) < 0.5
    for (fx, fy) in ((0, 0), (cells - 7, 0), (0, cells - 7)):
        bits[fy:fy + 7, fx:fx + 7] = False
        d.rectangle([qx + fx * cs, qy + fy * cs, qx + (fx + 7) * cs, qy + (fy + 7) * cs], fill=(0, 0, 0))
        d.rectangle([qx + (fx + 1) * cs, qy + (fy + 1) * cs, qx + (fx + 6) * cs, qy + (fy + 6) * cs], fill=(255, 255, 255))
        d.rectangle([qx + (fx + 2) * cs, qy + (fy + 2) * cs, qx + (fx + 5) * cs, qy + (fy + 5) * cs], fill=(0, 0, 0))
    for i in range(cells):
        for j in range(cells):
            if bits[j, i]:
                d.rectangle([qx + i * cs, qy + j * cs, qx + (i + 1) * cs, qy + (j + 1) * cs], fill=(0, 0, 0))
    # small logo mark top right
    lr = int(min(W, H) * 0.045)
    lx, ly = W - int(W * 0.04) - lr, int(H * 0.05)
    d.ellipse([lx - lr, ly - lr + lr, lx + lr, ly + lr + lr], outline=(255, 255, 255), width=max(2, lr // 6))
    text(d, (lx - lr * 3.2, ly + lr), "BRAND", F("sansB", max(12, lr)), (255, 255, 255), "mm")
    return im


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


def ad_portrait(kind):
    """Upright versions of the three ads for the column lightboxes (5:6)."""
    W, H = 1000, 1200
    if kind == 0:
        im = grad(W, H, (250, 120, 60), (250, 210, 120))
        d = ImageDraw.Draw(im)
        d.ellipse([430, 560, 930, 1060], fill=(255, 240, 200))
        d.rounded_rectangle([580, 600, 780, 1060], 40, fill=(255, 255, 255))
        d.rounded_rectangle([600, 660, 760, 1040], 30, fill=(250, 150, 60))
        text(d, (70, 130), "夏の果実、", F("maruK", 110), (255, 255, 255))
        text(d, (70, 280), "はじける", F("maruK", 110), (255, 255, 255))
        text(d, (70, 430), "ソーダ。", F("maruK", 110), (255, 255, 255))
        text(d, (70, 1110), "期間限定", F("sansB", 54), (120, 40, 10))
    elif kind == 1:
        im = grad(W, H, (20, 40, 70), (40, 90, 140))
        d = ImageDraw.Draw(im)
        for i in range(8):
            x = 120 + i * 100
            d.rectangle([x, 1150 - (i * 53) % 300 - 200, x + 78, 1150], fill=(15, 25, 45))
        text(d, (70, 140), "はじめての", F("sansB", 110), (255, 255, 255))
        text(d, (70, 290), "英会話、", F("sansB", 110), (255, 255, 255))
        text(d, (70, 440), "駅前で。", F("sansB", 110), (255, 255, 255))
        d.rounded_rectangle([70, 620, 930, 740], 20, fill=(255, 205, 0))
        text(d, (500, 680), "無料体験レッスン受付中", F("sansB", 62), (20, 30, 50), "mm")
    else:
        im = Image.new("RGB", (W, H), (245, 243, 238))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 620, W, H], fill=(40, 70, 60))
        for k in range(6):
            r = 230 - k * 38
            d.ellipse([500 - r, 910 - r, 500 + r, 910 + r], outline=(200, 220, 200), width=6)
        text(d, (80, 140), "暮らしに、", F("serifB", 120), (40, 60, 50))
        text(d, (80, 300), "緑をひとつ。", F("serifB", 120), (40, 60, 50))
        text(d, (85, 500), "インテリアグリーンフェア 開催中", F("sansM", 46), (80, 90, 80))
    return add_print_texture(im, 20 + kind, 2.0)


def portal_sign():
    """Direction sign on the portal at the foot of stair J (layout as in the 2022 photo)."""
    W, H = 4200, 300
    ORANGE = (236, 98, 36)
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 260, H], fill=ORANGE)
    text(d, (130, 150), "2", F("sansB", 230), (255, 255, 255), "mm")
    text(d, (310, 120), "新大阪・千里中央", F("sansB", 120), (25, 25, 25), "lm")
    text(d, (1290, 135), "方面", F("sansM", 70), (25, 25, 25), "lm")
    text(d, (315, 240), "for Shin-Osaka, Senri-chuo", F("sansM", 52), (50, 50, 50), "lm")
    d.rectangle([W - 400, 0, W - 140, H], fill=ORANGE)
    text(d, (W - 270, 150), "1", F("sansB", 230), (255, 255, 255), "mm")
    d.rectangle([W - 140, 0, W, H], fill=(30, 30, 30))
    arrow(d, W - 70, 150, 110, (255, 255, 255), "right")
    text(d, (W - 560, 120), "なんば・天王寺・なかもず", F("sansB", 120), (25, 25, 25), "rm")
    d.rectangle([W - 560, 40, W - 420, 200], fill=(250, 250, 250))
    text(d, (W - 548, 135), "方面", F("sansM", 70), (25, 25, 25), "lm")
    text(d, (W - 440, 240), "for Namba, Tennoji, Nakamozu", F("sansM", 52), (50, 50, 50), "rm")
    return im


def decal(color, up):
    W = 400
    im = Image.new("RGB", (W, W), (245, 245, 245))
    d = ImageDraw.Draw(im)
    arrow(d, W // 2, W // 2, 300, color, "up" if up else "down")
    return add_print_texture(im, 40 + int(up), 3.0)


def psd_label(track=2, direction="新大阪方面", car=1, door=3):
    im = Image.new("RGB", (360, 280), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 360, 70], fill=RED)
    text(d, (180, 36), f"{track}  {direction}", F("sansB", 40), (255, 255, 255), "mm")
    text(d, (180, 170), f"{car}-{door}", F("sansB", 120), (30, 30, 30), "mm")
    return im


PSD_ATLAS = (8, 5)        # columns, rows of 360 x 280 labels: car c door d at index (c-1)*4 + d-1


def psd_label_atlas(track, direction):
    """Every car / door label of one track in one image (the quads pick their cell by UV)."""
    cols, rows = PSD_ATLAS
    im = Image.new("RGB", (360 * cols, 280 * rows), (250, 250, 250))
    for c in range(1, 11):
        for dr in range(1, 5):
            i = (c - 1) * 4 + dr - 1
            im.paste(psd_label(track, direction, c, dr), ((i % cols) * 360, (i // cols) * 280))
    return im


def name_board2(left, right):
    """Station name board in the current Osaka Metro layout (wp_namesign photo): white, red band
    with the station numbers, neighbouring stations below; 'left'/'right' = (num, kanji, kana, roman)."""
    W, H = 2400, 950
    im = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(im)
    text(d, (W / 2, 95), "うめだ", F("sansM", 60), (40, 40, 40), "mm")
    text(d, (W / 2, 250), "梅田", F("sansB", 230), (20, 20, 20), "mm")
    text(d, (W / 2, 420), "Umeda", F("sansM", 80), (40, 40, 40), "mm")
    d.rectangle([0, 500, W, 580], fill=RED)
    text(d, (W / 2, 540), "M16", F("sansB", 66), (255, 255, 255), "mm")
    d.rectangle([0, 500, 190, 580], fill=(255, 255, 255)); d.rectangle([4, 504, 186, 576], fill=RED)
    d.rectangle([W - 190, 500, W, 580], fill=(255, 255, 255)); d.rectangle([W - 186, 504, W - 4, 576], fill=RED)
    text(d, (95, 540), left[0], F("sansB", 56), (255, 255, 255), "mm")
    text(d, (W - 95, 540), right[0], F("sansB", 56), (255, 255, 255), "mm")
    text(d, (W - 330, 540), "》", F("sansB", 70), (255, 255, 255), "mm")
    for (num, kan, kana, rom), x, anc in ((left, 60, "l"), (right, W - 60, "r")):
        text(d, (x, 640), kana, F("sansM", 44), (60, 60, 60), anc + "m")
        text(d, (x, 740), kan, F("sansB", 110), (25, 25, 25), anc + "m")
        text(d, (x, 860), rom, F("sansM", 52), (60, 60, 60), anc + "m")
    # line badge top right
    d.ellipse([W - 330, 60, W - 210, 180], fill=RED)
    text(d, (W - 270, 120), "M", F("sansB", 90), (255, 255, 255), "mm")
    text(d, (W - 270, 230), "御堂筋線", F("sansB", 44), (40, 40, 40), "mm")
    return im


def dep_board(track, rows):
    W, H = 1900, 550
    im = Image.new("RGB", (W, H), (8, 8, 10))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 170, H], fill=RED)
    text(d, (85, H / 2), str(track), F("sansB", 200), (255, 255, 255), "mm")
    text(d, (230, 40), "発車時刻 Departure Time", F("sansM", 34), (200, 200, 200), "lm")
    text(d, (1150, 40), "行先 Destination", F("sansM", 34), (200, 200, 200), "lm")
    for k, (tm, lab, dest, num) in enumerate(rows):
        y = 165 + k * 190
        text(d, (230, y), tm, F("sansB", 120), (255, 255, 255), "lm")
        text(d, (630, y - 18), lab, F("sansB", 56), (255, 120, 200) if k == 0 else (120, 200, 255), "lm")
        text(d, (1080, y), dest, F("sansB", 110), (255, 255, 255), "lm")
        d.rectangle([1640, y - 50, 1850, y + 50], fill=(230, 40, 60))
        text(d, (1745, y), num, F("sansB", 70), (255, 255, 255), "mm")
    text(d, (230, 500), "Osaka Metro ホームページをご覧ください", F("sansM", 40), (255, 170, 60), "lm")
    return im


def clock_face():
    W = 600
    im = Image.new("RGB", (W, W), (20, 20, 22))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 20, W - 20, W - 20], fill=(245, 245, 245))
    c = W / 2
    for k in range(60):
        a = math.radians(k * 6)
        r0 = 0.40 * W if k % 5 else 0.34 * W
        d.line([(c + r0 * math.sin(a), c - r0 * math.cos(a)), (c + 0.45 * W * math.sin(a), c - 0.45 * W * math.cos(a))], fill=(20, 20, 20), width=6 if k % 5 == 0 else 3)
    for ang, ln, wd in ((math.radians(11 * 30 + 28), 0.24, 16), (math.radians(56 * 6), 0.38, 10)):
        d.line([(c, c), (c + ln * W * math.sin(ang), c - ln * W * math.cos(ang))], fill=(20, 20, 20), width=wd)
    d.ellipse([c - 14, c - 14, c + 14, c + 14], fill=(20, 20, 20))
    return im


def queue_mark(kind):
    """Floor graphic in front of a platform door: chevrons showing where to queue (fictional pattern)."""
    W, H = 900, 650
    base = (236, 214, 205) if kind == 0 else (205, 220, 236)
    ink = (200, 70, 60) if kind == 0 else (50, 100, 190)
    im = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(im)
    for k in range(5):
        y = 60 + k * 120
        d.polygon([(250, y), (450, y + 60), (650, y), (650, y + 35), (450, y + 95), (250, y + 35)], fill=ink)
    d.rectangle([0, 0, W, 26], fill=ink); d.rectangle([0, H - 26, W, H], fill=ink)
    return add_print_texture(im, 50 + kind, 3.0)


def b1_sign_down():
    """Black hanging sign in the north-gate corridor: elevator to the left, Midosuji Line platforms down."""
    W, H = 4200, 620
    im = Image.new("RGB", (W, H), (22, 22, 24))
    d = ImageDraw.Draw(im)
    arrow(d, 200, 310, 260, (255, 255, 255), "left")
    d.rounded_rectangle([380, 190, 620, 430], 24, outline=(255, 255, 255), width=14)
    text(d, (500, 310), "EV", F("sansB", 90), (255, 255, 255), "mm")
    text(d, (720, 310), "c", F("sansB", 200), (255, 255, 255), "mm")
    d.line([(1150, 80), (1150, 540)], fill=(120, 120, 120), width=6)
    arrow(d, 1600, 310, 300, (255, 255, 255), "down")
    d.ellipse([1850, 180, 2110, 440], fill=RED)
    text(d, (1980, 310), "M", F("sansB", 190), (255, 255, 255), "mm")
    text(d, (2200, 250), "御堂筋線", F("sansB", 200), (255, 255, 255), "lm")
    text(d, (2210, 470), "Midosuji Line", F("sansM", 100), (230, 230, 230), "lm")
    return im


def gate_fascia_n():
    """Sign on the black fascia above the north gate: station/gate name and the line."""
    W, H = 3600, 400
    im = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 90, 560, 310], fill=(30, 70, 190))
    text(d, (310, 200), "Osaka Metro", F("sansB", 70), (255, 255, 255), "mm")
    text(d, (640, 160), "梅田駅 北改札", F("sansB", 140), (255, 255, 255), "lm")
    text(d, (645, 300), "Umeda Sta., Kita Gate", F("sansM", 70), (225, 225, 225), "lm")
    d.rounded_rectangle([1900, 90, 2050, 310], 20, fill=(236, 98, 36))
    text(d, (1975, 200), "M", F("sansB", 120), (255, 255, 255), "mm")
    text(d, (2110, 160), "御堂筋線", F("sansB", 140), (255, 255, 255), "lm")
    text(d, (2115, 300), "Midosuji Line", F("sansM", 70), (225, 225, 225), "lm")
    return im


def exit_green():
    """Emergency exit sign (green, running figure pictogram drawn with simple shapes)."""
    W, H = 600, 220
    im = Image.new("RGB", (W, H), (0, 150, 70))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 200, 200], fill=(255, 255, 255))
    d.rectangle([60, 40, 180, 200], fill=(0, 150, 70))
    d.ellipse([105, 60, 135, 90], fill=(255, 255, 255))
    d.polygon([(118, 95), (140, 130), (165, 150), (155, 160), (128, 140), (115, 165), (135, 195), (120, 200), (95, 170), (90, 140), (70, 150), (65, 138), (95, 115)], fill=(255, 255, 255))
    text(d, (400, 85), "非常口", F("sansB", 90), (255, 255, 255), "mm")
    text(d, (400, 170), "EXIT", F("sansB", 60), (255, 255, 255), "mm")
    return im


def psd_sticker():
    """Door caution sticker (generic wording and pictogram)."""
    W, H = 600, 300
    im = Image.new("RGB", (W, H), (255, 214, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, W - 8, H - 8], outline=(20, 20, 20), width=6)
    d.rectangle([30, 40, 150, 260], outline=(20, 20, 20), width=8)
    d.rectangle([150, 40, 270, 260], outline=(20, 20, 20), width=8)
    d.ellipse([125, 120, 175, 170], fill=(220, 30, 30))
    text(d, (290, 100), "ドアに", F("sansB", 54), (20, 20, 20), "lm")
    text(d, (290, 165), "手をはさまれない", F("sansB", 40), (20, 20, 20), "lm")
    text(d, (290, 220), "よう ご注意ください", F("sansB", 34), (20, 20, 20), "lm")
    return add_print_texture(im, 61, 2.0)


def ic_reader():
    W, H = 400, 300
    im = Image.new("RGB", (W, H), (20, 60, 170))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 20, W - 20, H - 20], 30, outline=(160, 200, 255), width=6)
    for k in range(3):
        d.arc([150 - 40 * k, 90 - 40 * k, 250 + 40 * k, 190 + 40 * k], 200, 340, fill=(220, 235, 255), width=8)
    text(d, (W / 2, 230), "IC", F("sansB", 70), (255, 255, 255), "mm")
    return im


def gate_indicator(ok):
    W = 200
    im = Image.new("RGB", (W, W), (10, 10, 12))
    d = ImageDraw.Draw(im)
    if ok:
        arrow(d, W // 2, W // 2, 150, (60, 255, 110), "up")
    else:
        d.line([(45, 45), (155, 155)], fill=(255, 50, 40), width=26)
        d.line([(155, 45), (45, 155)], fill=(255, 50, 40), width=26)
    return im


def ic_only_decal():
    W, H = 700, 300
    im = Image.new("RGB", (W, H), (60, 70, 170))
    d = ImageDraw.Draw(im)
    text(d, (160, 115), "IC", F("sansB", 120), (255, 255, 255), "mm")
    text(d, (160, 220), "専用", F("sansB", 70), (255, 255, 255), "mm")
    arrow(d, 520, 150, 220, (255, 255, 255), "up")
    text(d, (350, 270), "IC Card Only", F("sansM", 34), (230, 230, 255), "mm")
    return add_print_texture(im, 62, 3.0)


def hydrant_sign():
    W, H = 500, 160
    im = Image.new("RGB", (W, H), (210, 25, 30))
    d = ImageDraw.Draw(im)
    text(d, (W / 2, 70), "消火栓", F("sansB", 90), (255, 255, 255), "mm")
    text(d, (W / 2, 135), "FIRE HYDRANT", F("sansM", 34), (255, 255, 255), "mm")
    return im


def fire_ext_sign():
    W, H = 450, 150
    im = Image.new("RGB", (W, H), (210, 25, 30))
    d = ImageDraw.Draw(im)
    text(d, (W / 2, 66), "消火器", F("sansB", 86), (255, 255, 255), "mm")
    text(d, (W / 2, 128), "FIRE EXTINGUISHER", F("sansM", 28), (255, 255, 255), "mm")
    return im


def aed_sign():
    W, H = 400, 300
    im = Image.new("RGB", (W, H), (0, 140, 80))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 50, 200, 210], fill=(255, 255, 255))
    d.polygon([(120, 80), (95, 140), (125, 140), (105, 190), (160, 115), (130, 115), (150, 80)], fill=(220, 30, 30))
    text(d, (290, 120), "AED", F("sansB", 90), (255, 255, 255), "mm")
    text(d, (290, 220), "自動体外式除細動器", F("sansM", 22), (255, 255, 255), "mm")
    return im


def emergency_button():
    W, H = 300, 400
    im = Image.new("RGB", (W, H), (250, 210, 0))
    d = ImageDraw.Draw(im)
    text(d, (W / 2, 60), "非常停止", F("sansB", 56), (20, 20, 20), "mm")
    text(d, (W / 2, 120), "ボタン", F("sansB", 56), (20, 20, 20), "mm")
    d.ellipse([90, 180, 210, 300], fill=(30, 30, 30))
    d.ellipse([105, 195, 195, 285], fill=(225, 30, 30))
    text(d, (W / 2, 350), "EMERGENCY STOP", F("sansM", 26), (20, 20, 20), "mm")
    return im


def asset_label(n):
    im = Image.new("RGB", (300, 100), (235, 235, 235))
    d = ImageDraw.Draw(im)
    text(d, (150, 35), f"UMD-DS-{n:02d}", F("sansM", 34), (40, 40, 40), "mm")
    text(d, (150, 75), "管理番号 / 取扱注意", F("sansM", 20), (60, 60, 60), "mm")
    return im


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
        "portal_sign": portal_sign(),
        "decal_up": decal((220, 40, 40), True),
        "decal_down": decal((40, 110, 210), False),
        "psd_label": psd_label(),
        "psd_label_1": psd_label_atlas(1, "なんば方面"),
        "psd_label_2": psd_label_atlas(2, "新大阪方面"),
        "fire_ext": fire_ext_sign(),
        "name_board_w": name_board2(("M17", "淀屋橋", "よどやばし", "Yodoyabashi"), ("M15", "中津", "なかつ", "Nakatsu")),
        "name_board_e": name_board2(("M15", "中津", "なかつ", "Nakatsu"), ("M17", "淀屋橋", "よどやばし", "Yodoyabashi")),
        "dep1": dep_board(1, [("11:56", "先発", "なかもず", "M30"), ("11:58", "次発", "天王寺", "M23")]),
        "dep2": dep_board(2, [("19:30", "先発", "千里中央", "M08"), ("19:34", "次発", "新大阪", "M13")]),
        "clock": clock_face(),
        "b1_sign_down": b1_sign_down(),
        "gate_fascia_n": gate_fascia_n(),
        "exit_green": exit_green(),
        "psd_sticker": psd_sticker(), "ic_reader": ic_reader(),
        "gate_ok": gate_indicator(True), "gate_ng": gate_indicator(False),
        "ic_only": ic_only_decal(), "hydrant": hydrant_sign(), "aed": aed_sign(),
        "estop": emergency_button(), "asset": asset_label(3),
        "queue0": queue_mark(0), "queue1": queue_mark(1),
        "led_vision": led_campaign(),
        "band0": enrich(band_ad(0), 7), "band1": enrich(band_ad(1), 8), "band2": enrich(band_ad(2), 9),
        "ad0": enrich(ad_panel(0), 1), "ad1": enrich(ad_panel(1), 2), "ad2": enrich(ad_panel(2), 3),
        "adp0": enrich(ad_portrait(0), 4), "adp1": enrich(ad_portrait(1), 5), "adp2": enrich(ad_portrait(2), 6),
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
