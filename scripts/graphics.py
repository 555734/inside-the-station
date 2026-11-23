"""2D graphics (posters, signs, plates) drawn with Pillow. All designs are original.

Fictional operator: 陽ヶ丘地下鉄 (Higaoka Metro).
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.join(os.path.dirname(__file__), "..")
FONTS = os.path.join(ROOT, "fonts")
OUT = os.path.join(ROOT, "cache", "gfx")


def F(name, size):
    table = {
        "sans": "NotoSansJP_400Regular.ttf", "sansM": "NotoSansJP_500Medium.ttf",
        "sansB": "NotoSansJP_700Bold.ttf", "sansK": "NotoSansJP_900Black.ttf",
        "sansL": "NotoSansJP_300Light.ttf",
        "serif": "NotoSerifJP_500Medium.ttf", "serifB": "NotoSerifJP_700Bold.ttf",
        "serifK": "NotoSerifJP_900Black.ttf", "serifL": "NotoSerifJP_300Light.ttf",
        "maru": "ZenMaruGothic_700Bold.ttf", "maruK": "ZenMaruGothic_900Black.ttf",
        "maruM": "ZenMaruGothic_500Medium.ttf",
        "dela": "DelaGothicOne_400Regular.ttf", "mplusK": "MPLUS1p_900Black.ttf",
        "mplusB": "MPLUS1p_700Bold.ttf",
    }
    return ImageFont.truetype(os.path.join(FONTS, table[name]), size)


def text(d, xy, s, font, fill, anchor="la", spacing=0):
    if spacing:
        x, y = xy
        total = sum(d.textlength(ch, font=font) for ch in s) + spacing * (len(s) - 1)
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        anchor = "l" + anchor[1]
        for ch in s:
            d.text((x, y), ch, font=font, fill=fill, anchor=anchor)
            x += d.textlength(ch, font=font) + spacing
        return
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def vtext(d, x, y, s, font, fill, step=None):
    step = step or font.size * 1.02
    for i, ch in enumerate(s):
        d.text((x, y + i * step), ch, font=font, fill=fill, anchor="mt")


def finelines(d, x, y, w, lines, font, fill, lh):
    for i, s in enumerate(lines):
        d.text((x, y + i * lh), s, font=font, fill=fill)


def grad(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, np.float32)[None, None]
    b = np.array(bottom, np.float32)[None, None]
    g = (a * (1 - t) + b * t).repeat(w, 1)
    return Image.fromarray(g.clip(0, 255).astype(np.uint8), "RGB")


def noise_img(w, h, seed, amp=6):
    r = np.random.default_rng(seed).normal(0, amp, (h, w, 1)).repeat(3, 2)
    return r


def add_print_texture(im, seed, amp=3.5):
    a = np.asarray(im).astype(np.float32)
    a += noise_img(im.width, im.height, seed, amp)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


PW, PH = 1092, 1545  # B1 portrait @1.5 px/mm


# ------------------------------------------------------------------ posters

def poster_mansion():
    im = grad(PW, PH, (120, 170, 220), (226, 236, 244))
    d = ImageDraw.Draw(im)
    # clouds
    rng = np.random.default_rng(3)
    cl = Image.new("L", (PW, PH), 0)
    cd = ImageDraw.Draw(cl)
    for _ in range(40):
        x, y = rng.uniform(-100, PW), rng.uniform(150, 650)
        r = rng.uniform(30, 90)
        cd.ellipse([x - r * 2, y - r, x + r * 2, y + r], fill=int(rng.uniform(60, 140)))
    cl = cl.filter(ImageFilter.GaussianBlur(28))
    im = Image.composite(Image.new("RGB", (PW, PH), (250, 252, 255)), im, cl)
    d = ImageDraw.Draw(im)
    # tower
    bx0, bx1, by0 = 300, 720, 330
    d.polygon([(bx0, PH - 380), (bx0, by0 + 40), (bx0 + 60, by0), (bx1, by0), (bx1, PH - 380)], fill=(214, 220, 228))
    d.polygon([(bx1, by0), (bx1 + 120, by0 + 30), (bx1 + 120, PH - 380), (bx1, PH - 380)], fill=(160, 172, 188))
    for yy in range(by0 + 30, PH - 400, 26):
        for xx in range(bx0 + 14, bx1 - 10, 34):
            c = (110, 140, 175) if rng.random() > 0.15 else (240, 214, 150)
            d.rectangle([xx, yy, xx + 22, yy + 15], fill=c)
        d.line([(bx0, yy + 20), (bx1, yy + 20)], fill=(235, 238, 242), width=3)
        for xx in range(bx1 + 8, bx1 + 115, 28):
            d.rectangle([xx, yy + 4, xx + 18, yy + 16], fill=(90, 110, 140))
    # low buildings / trees
    for i in range(14):
        x = i * 85 - 30
        h = rng.uniform(80, 200)
        d.rectangle([x, PH - 380 - h, x + 80, PH - 380], fill=(130 + i % 3 * 10, 140, 150))
    for i in range(30):
        x = rng.uniform(0, PW); r = rng.uniform(25, 55)
        d.ellipse([x - r, PH - 420 - r * 0.6, x + r, PH - 380 + r * 0.2], fill=(70, 110 + int(rng.uniform(0, 30)), 70))
    d.rectangle([0, PH - 385, PW, PH], fill=(255, 255, 255))
    d.rectangle([0, 0, PW, 110], fill=(25, 42, 74))
    text(d, (PW / 2, 58), "陽ヶ丘線「中央公園」駅 直結・徒歩1分", F("sansB", 40), (255, 255, 255), "mm")
    vtext(d, PW - 95, 160, "空に、", F("serifB", 76), (25, 42, 74))
    vtext(d, PW - 185, 240, "いちばん近い", F("serifB", 76), (25, 42, 74))
    vtext(d, PW - 275, 160, "わが家。", F("serifB", 76), (25, 42, 74))
    text(d, (70, PH - 345), "GRAND FORT", F("serifL", 42), (150, 120, 60), spacing=10)
    text(d, (70, PH - 290), "グランフォート中央公園", F("serifB", 78), (25, 42, 74))
    d.rectangle([70, PH - 180, 520, PH - 120], fill=(180, 140, 60))
    text(d, (295, PH - 150), "モデルルーム公開中", F("sansB", 38), (255, 255, 255), "mm")
    text(d, (560, PH - 182), "全212邸・地上34階", F("sansB", 34), (25, 42, 74))
    text(d, (560, PH - 138), "2LDK〜4LDK", F("sansM", 30), (25, 42, 74))
    finelines(d, 70, PH - 95, PW, [
        "■お問い合わせ グランフォート中央公園 販売準備室 0120-000-000 (10:00〜18:00 水・木定休)",
        "※掲載の完成予想図は計画段階の図面を基に描き起こしたもので、実際とは異なります。",
    ], F("sans", 19), (90, 90, 90), 28)
    return add_print_texture(im, 1)


def poster_manners():
    im = Image.new("RGB", (PW, PH), (255, 214, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PW, 140], fill=(30, 30, 30))
    text(d, (PW / 2, 72), "駅構内でのお願い", F("sansB", 58), (255, 214, 0), "mm", spacing=8)
    text(d, (PW / 2, 260), "歩きスマホは、", F("maruK", 120), (30, 30, 30), "mm")
    text(d, (PW / 2, 400), "やめましょう。", F("maruK", 120), (30, 30, 30), "mm")
    # illustration: person looking at phone about to bump into another person
    cx, cy = 360, 900
    k = (30, 30, 30)
    d.ellipse([cx - 55, cy - 290, cx + 55, cy - 180], fill=k)                    # head (tilted down)
    d.rounded_rectangle([cx - 70, cy - 170, cx + 70, cy + 60], 40, fill=k)       # torso
    d.line([(cx - 30, cy + 40), (cx - 90, cy + 260)], fill=k, width=48)          # legs
    d.line([(cx + 30, cy + 40), (cx + 70, cy + 260)], fill=k, width=48)
    d.line([(cx + 40, cy - 130), (cx + 130, cy - 120)], fill=k, width=40)        # arm to phone
    d.rounded_rectangle([cx + 120, cy - 175, cx + 160, cy - 100], 8, fill=(255, 255, 255), outline=k, width=8)
    ox = 780
    d.ellipse([ox - 55, cy - 300, ox + 55, cy - 190], fill=(200, 40, 40))
    d.rounded_rectangle([ox - 70, cy - 180, ox + 70, cy + 60], 40, fill=(200, 40, 40))
    d.line([(ox - 30, cy + 40), (ox - 60, cy + 260)], fill=(200, 40, 40), width=48)
    d.line([(ox + 30, cy + 40), (ox + 80, cy + 260)], fill=(200, 40, 40), width=48)
    for a in (-30, 0, 30):
        r = math.radians(a)
        x0, y0 = 570 + 40 * math.sin(r), cy - 240 - 40 * math.cos(r)
        x1, y1 = 570 + 110 * math.sin(r), cy - 240 - 110 * math.cos(r)
        d.line([(x0, y0), (x1, y1)], fill=k, width=14)
    text(d, (PW / 2, 1240), "ぶつかる・転ぶ・ホームから落ちる。", F("sansB", 46), k, "mm")
    text(d, (PW / 2, 1310), "画面ではなく、まわりを見て歩きましょう。", F("sansM", 38), k, "mm")
    d.rectangle([0, PH - 130, PW, PH], fill=(255, 255, 255))
    text(d, (70, PH - 65), "陽ヶ丘地下鉄", F("sansB", 44), k, "lm")
    text(d, (PW - 70, PH - 65), "HIGAOKA METRO", F("sansB", 34), (0, 110, 160), "rm", spacing=4)
    return add_print_texture(im, 2)


def poster_museum():
    W, H = PW, PH
    im = grad(W, H, (232, 226, 210), (214, 205, 186))
    a = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(7)
    # layered mountains with mist (sumi-e like)
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    for i, (base, amp, tone) in enumerate([(520, 140, 150), (640, 120, 115), (760, 110, 80), (880, 90, 50)]):
        ridge = base + sum(amp / (j + 1) * np.sin(x / (230 / (j + 1)) + rng.uniform(0, 6)) for j in range(4))
        mask = np.clip((y - ridge) / 6.0, 0, 1)
        mist = np.clip(1 - (y - ridge) / 260.0, 0.0, 1.0)
        col = tone + (225 - tone) * (1 - mist) * 0.55
        a = a * (1 - mask[..., None]) + col[..., None] * mask[..., None] * np.array([1.0, 0.98, 0.93])
    # red sun
    sx, sy, sr = 760, 360, 95
    sun = np.clip((sr - np.sqrt((x - sx) ** 2 + (y - sy) ** 2)) / 2.0, 0, 1)
    a = a * (1 - sun[..., None]) + np.array([196, 52, 40]) * sun[..., None]
    paper = ndimage_noise(W, H, 9)
    a *= (1 + 0.035 * paper)[..., None]
    a[1000:] = np.array([28, 28, 30])
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    vtext(d, 140, 110, "日本画の百年", F("serifK", 110), (25, 25, 25), 116)
    vtext(d, 250, 140, "山と霧のあいだ", F("serifB", 44), (60, 50, 45), 48)
    text(d, (70, 1050), "NIHONGA: ONE HUNDRED YEARS", F("serifL", 36), (220, 210, 190), spacing=4)
    text(d, (70, 1120), "2026.10.3", F("serifB", 96), (240, 235, 225))
    text(d, (560, 1150), "土", F("serif", 40), (240, 235, 225))
    text(d, (620, 1125), "—", F("serif", 70), (240, 235, 225))
    text(d, (700, 1120), "12.20", F("serifB", 96), (240, 235, 225))
    text(d, (970, 1150), "日", F("serif", 40), (240, 235, 225))
    d.line([(70, 1260), (PW - 70, 1260)], fill=(120, 110, 100), width=2)
    text(d, (70, 1290), "陽ヶ丘市立美術館", F("serifB", 56), (240, 235, 225))
    finelines(d, 70, 1380, PW, [
        "開館時間 9:30〜17:00（金曜は20:00まで）  休館日 月曜日（祝日の場合は翌日）",
        "観覧料 一般 1,600円 / 大学生 1,000円 / 高校生以下無料",
        "陽ヶ丘線「美術館前」駅 2番出口より徒歩3分",
    ], F("sans", 24), (200, 195, 185), 40)
    return add_print_texture(im, 3, 2.5)


def ndimage_noise(W, H, seed):
    from scipy import ndimage
    n = np.random.default_rng(seed).standard_normal((H, W)).astype(np.float32)
    n = ndimage.gaussian_filter(n, 2.0) * 4 + ndimage.gaussian_filter(n, 30) * 40
    return n / (np.abs(n).max() + 1e-6)


def poster_ramen():
    im = Image.new("RGB", (PW, PH), (22, 18, 16))
    d = ImageDraw.Draw(im)
    cx, cy = PW / 2, 820
    # bowl (top view-ish ellipse)
    d.ellipse([cx - 430, cy - 300, cx + 430, cy + 300], fill=(140, 30, 25))
    d.ellipse([cx - 400, cy - 275, cx + 400, cy + 275], fill=(205, 150, 80))
    soup = Image.new("L", (PW, PH), 0)
    ImageDraw.Draw(soup).ellipse([cx - 380, cy - 255, cx + 380, cy + 255], fill=255)
    rng = np.random.default_rng(11)
    layer = Image.new("RGB", (PW, PH), (196, 132, 58))
    ld = ImageDraw.Draw(layer)
    for _ in range(130):   # noodles
        x0 = rng.uniform(cx - 360, cx + 360); y0 = rng.uniform(cy - 60, cy + 230)
        pts = [(x0 + k * 16, y0 + 14 * math.sin(k / 2 + x0)) for k in range(int(rng.uniform(5, 14)))]
        ld.line(pts, fill=(240, 205, 120), width=9)
    for _ in range(60):    # oil droplets
        x0 = rng.uniform(cx - 360, cx + 360); y0 = rng.uniform(cy - 240, cy + 240); r = rng.uniform(4, 14)
        ld.ellipse([x0 - r, y0 - r * 0.7, x0 + r, y0 + r * 0.7], fill=(230, 170, 70))
    # chashu, egg, negi, nori
    ld.ellipse([cx - 330, cy - 200, cx - 90, cy - 20], fill=(150, 80, 50))
    ld.ellipse([cx - 300, cy - 180, cx - 120, cy - 40], fill=(205, 150, 120))
    ld.ellipse([cx + 60, cy - 210, cx + 230, cy - 60], fill=(250, 245, 230))
    ld.ellipse([cx + 105, cy - 175, cx + 185, cy - 100], fill=(240, 150, 30))
    ld.polygon([(cx + 240, cy - 230), (cx + 360, cy - 160), (cx + 300, cy + 40), (cx + 180, cy - 30)], fill=(25, 35, 25))
    for _ in range(45):
        x0 = rng.uniform(cx - 120, cx + 60); y0 = rng.uniform(cy - 160, cy + 20); r = rng.uniform(7, 13)
        ld.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=(90, 160, 60), outline=(200, 230, 170), width=3)
    im.paste(layer, (0, 0), soup)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PW, 380], fill=(178, 24, 24))
    vtext(d, PW - 110, 40, "濃厚", F("dela", 130), (255, 240, 220), 132)
    text(d, (70, 70), "北海道の白味噌と", F("mplusB", 44), (255, 230, 210))
    text(d, (70, 130), "三日炊いた豚骨。", F("mplusB", 44), (255, 230, 210))
    text(d, (70, 220), "味噌らーめん", F("dela", 108), (255, 255, 255))
    text(d, (PW / 2, 1200), "麺処 ひだまり", F("dela", 110), (255, 255, 255), "mm")
    d.rounded_rectangle([110, 1300, PW - 110, 1400], 50, fill=(255, 214, 0))
    text(d, (PW / 2, 1350), "A2出口 出てすぐ 徒歩1分", F("sansK", 52), (22, 18, 16), "mm")
    text(d, (PW / 2, 1465), "営業時間 11:00〜翌2:00  年中無休", F("sansB", 36), (230, 220, 210), "mm")
    return add_print_texture(im, 4)


def poster_travel():
    im = grad(PW, PH, (60, 150, 210), (190, 228, 245))
    a = np.asarray(im).astype(np.float32)
    H = PH
    hz = 700
    sea = grad(PW, H - hz, (20, 110, 170), (90, 180, 205))
    a[hz:] = np.asarray(sea).astype(np.float32)
    rng = np.random.default_rng(5)
    for _ in range(900):   # sparkles on water
        x = int(rng.uniform(0, PW)); y = int(rng.uniform(hz + 5, 1150)); L = int(rng.uniform(4, 20))
        a[y:y + 2, x:x + L] = a[y:y + 2, x:x + L] * 0.5 + 255 * 0.5
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # beach & train on viaduct
    d.polygon([(0, 1150), (PW, 1060), (PW, PH), (0, PH)], fill=(236, 220, 186))
    d.rectangle([0, 980, PW, 1000], fill=(200, 200, 205))
    for xx in range(40, PW, 160):
        d.rectangle([xx, 1000, xx + 22, 1110], fill=(185, 185, 190))
    d.rounded_rectangle([140, 905, 980, 982], 30, fill=(245, 245, 245))
    d.rectangle([140, 950, 980, 962], fill=(0, 120, 170))
    for xx in range(200, 960, 70):
        d.rectangle([xx, 920, xx + 46, 942], fill=(60, 80, 100))
    d.rectangle([0, 0, PW, 60], fill=(255, 255, 255))
    text(d, (PW / 2, 32), "陽ヶ丘地下鉄 × 潮見ライン  週末きっぷ", F("sansB", 30), (0, 90, 140), "mm")
    text(d, (80, 120), "週末は、", F("maruK", 130), (255, 255, 255))
    text(d, (80, 270), "海に会いに。", F("maruK", 130), (255, 255, 255))
    text(d, (86, 440), "中央公園から乗りかえ1回、90分。", F("sansB", 44), (255, 255, 255))
    d.rounded_rectangle([80, 1210, 640, 1330], 18, fill=(0, 90, 140))
    text(d, (110, 1270), "往復フリーきっぷ", F("sansB", 44), (255, 255, 255), "lm")
    text(d, (690, 1250), "大人", F("sansB", 36), (0, 70, 110))
    text(d, (770, 1220), "2,400", F("sansK", 86), (0, 70, 110))
    text(d, (1010, 1260), "円", F("sansB", 36), (0, 70, 110))
    finelines(d, 80, 1380, PW, [
        "発売期間 2026年11月1日〜2027年3月31日  各駅の券売機・窓口で発売",
        "※ご利用当日を含む2日間有効。一部の特急列車にはご乗車いただけません。",
    ], F("sans", 22), (90, 80, 60), 36)
    return add_print_texture(im, 5)


def poster_notice():
    im = Image.new("RGB", (PW, PH), (250, 250, 248))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PW, 300], fill=(200, 25, 35))
    text(d, (PW / 2, 150), "ご注意ください", F("sansK", 110), (255, 255, 255), "mm", spacing=10)
    text(d, (PW / 2, 470), "不審なもの・", F("sansK", 104), (30, 30, 30), "mm")
    text(d, (PW / 2, 600), "不審な人物を", F("sansK", 104), (30, 30, 30), "mm")
    text(d, (PW / 2, 730), "見かけたら", F("sansK", 104), (30, 30, 30), "mm")
    d.rounded_rectangle([90, 820, PW - 90, 1000], 24, outline=(200, 25, 35), width=10)
    text(d, (PW / 2, 910), "駅係員・警察官へ", F("sansK", 96), (200, 25, 35), "mm")
    text(d, (PW / 2, 1080), "お知らせください。", F("sansB", 70), (30, 30, 30), "mm")
    text(d, (PW / 2, 1200), "If you see any suspicious object or person,", F("sansM", 34), (80, 80, 80), "mm")
    text(d, (PW / 2, 1250), "please notify station staff or police.", F("sansM", 34), (80, 80, 80), "mm")
    d.rectangle([0, PH - 150, PW, PH], fill=(30, 30, 30))
    text(d, (PW / 2, PH - 75), "陽ヶ丘地下鉄 ・ 陽ヶ丘中央警察署", F("sansB", 42), (255, 255, 255), "mm")
    return add_print_texture(im, 6)


# ------------------------------------------------------------------ signs

def arrow(d, cx, cy, size, fill, direction="up"):
    s = size
    pts = [(0, -0.5), (0.42, -0.02), (0.15, -0.02), (0.15, 0.5), (-0.15, 0.5), (-0.15, -0.02), (-0.42, -0.02)]
    if direction == "left":
        pts = [(y, -x) for x, y in pts]
    if direction == "right":
        pts = [(-y, x) for x, y in pts]
    d.polygon([(cx + x * s, cy + y * s) for x, y in pts], fill=fill)


def sign_exit():
    W, H = 3000, 600   # 1.80 x 0.36 m
    im = Image.new("RGB", (W, H), (255, 205, 0))
    d = ImageDraw.Draw(im)
    k = (20, 20, 20)
    arrow(d, 250, 300, 330, k, "up")
    text(d, (470, 300), "出口", F("sansB", 250), k, "lm")
    text(d, (1000, 330), "A", F("sansB", 300), k, "lm")
    text(d, (1210, 330), "2", F("sansB", 300), k, "lm")
    text(d, (490, 520), "Exit", F("sansM", 92), k, "lm")
    d.rectangle([1440, 90, 1452, 510], fill=k)
    text(d, (1530, 210), "中央公園・市役所", F("sansB", 150), k, "lm")
    text(d, (1530, 420), "Chuo Park · City Hall", F("sansM", 100), k, "lm")
    return im


def sign_transfer():
    W, H = 2600, 520   # 1.50 x 0.30 m
    im = Image.new("RGB", (W, H), (34, 36, 40))
    d = ImageDraw.Draw(im)
    w = (250, 250, 250)
    arrow(d, 230, 260, 300, w, "left")
    d.ellipse([430, 120, 710, 400], fill=(0, 150, 120))
    d.ellipse([470, 160, 670, 360], fill=(255, 255, 255))
    text(d, (570, 262), "S", F("sansB", 150), (0, 150, 120), "mm")
    text(d, (780, 200), "潮見ライン", F("sansB", 170), w, "lm")
    text(d, (790, 400), "Shiomi Line", F("sansM", 100), w, "lm")
    text(d, (1820, 200), "のりかえ", F("sansB", 140), w, "lm")
    text(d, (1830, 400), "Transfer", F("sansM", 100), w, "lm")
    return im


def sign_emergency():
    W, H = 900, 300
    im = Image.new("RGB", (W, H), (245, 250, 245))
    d = ImageDraw.Draw(im)
    g = (0, 150, 70)
    d.rectangle([0, 0, 300, H], fill=g)
    # running figure
    w = (250, 255, 250)
    d.ellipse([170, 40, 215, 85], fill=w)
    d.line([(165, 100), (120, 190)], fill=w, width=34)
    d.line([(120, 190), (175, 250)], fill=w, width=26)
    d.line([(120, 190), (60, 225)], fill=w, width=26)
    d.line([(160, 110), (220, 140)], fill=w, width=22)
    d.line([(155, 110), (95, 120)], fill=w, width=22)
    d.rectangle([230, 60, 285, 260], outline=w, width=10)
    arrow(d, 600, 150, 230, g, "left")
    return im


def plate_staff():
    W, H = 600, 240
    im = Image.new("RGB", (W, H), (248, 248, 246))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 70], fill=(200, 30, 40))
    text(d, (W / 2, 36), "STAFF ONLY", F("sansB", 44), (255, 255, 255), "mm", spacing=4)
    text(d, (W / 2, 125), "関係者以外", F("sansB", 66), (30, 30, 30), "mm")
    text(d, (W / 2, 200), "立入禁止", F("sansB", 66), (30, 30, 30), "mm")
    return im


def plate_hydrant():
    W, H = 800, 1100
    im = Image.new("RGB", (W, H), (190, 22, 26))
    d = ImageDraw.Draw(im)
    d.rectangle([30, 30, W - 30, H - 30], outline=(150, 14, 18), width=6)
    vtext(d, W / 2, 150, "消火栓", F("sansK", 200), (255, 255, 255), 230)
    text(d, (W / 2, 900), "HYDRANT", F("sansB", 70), (255, 255, 255), "mm", spacing=8)
    d.rectangle([W - 120, H / 2 - 60, W - 90, H / 2 + 60], fill=(120, 120, 120))
    return im


def wall_direction():
    W, H = 1600, 400
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    k = (25, 25, 25)
    arrow(d, 190, 200, 260, k, "left")
    text(d, (360, 150), "中央改札", F("sansB", 160), k, "lm")
    text(d, (370, 315), "Central Gate", F("sansM", 92), k, "lm")
    d.rectangle([1100, 90, 1112, 310], fill=k)
    text(d, (1160, 150), "バスのりば", F("sansB", 78), k, "lm")
    text(d, (1175, 300), "Bus", F("sansM", 80), k, "lm")
    return im


ALL = {
    "poster_mansion": poster_mansion, "poster_manners": poster_manners,
    "poster_museum": poster_museum, "poster_ramen": poster_ramen,
    "poster_travel": poster_travel, "poster_notice": poster_notice,
    "sign_exit": sign_exit, "sign_transfer": sign_transfer,
    "sign_emergency": sign_emergency, "plate_staff": plate_staff,
    "plate_hydrant": plate_hydrant, "wall_direction": wall_direction,
}


def build_all():
    os.makedirs(OUT, exist_ok=True)
    paths = {}
    for name, fn in ALL.items():
        p = os.path.join(OUT, name + ".png")
        fn().save(p)
        paths[name] = p
    return paths


if __name__ == "__main__":
    for k, v in build_all().items():
        print(k, v)
