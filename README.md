# inside-the-station

日本の地下鉄駅にある白タイルの連絡通路を、Blender（Cycles パストレーサ）でコードから生成した静止画です。
空間・素材・照明・ポスター・看板はすべてこのリポジトリのスクリプトで作っています。
3Dモデル素材、テクスチャ素材、写真は使っていません。
ポスターと看板は架空の「陽ヶ丘地下鉄」のものとしてデザインしたオリジナルです。

![render](render/corridor.jpg)

## 構成

| ファイル | 内容 |
|---|---|
| `scripts/textures.py` | タイル・床・天井・点字ブロックの質感（色・粗さ・法線）を numpy で生成します。タイルの丸い縁、1枚ごとの傾きや色のばらつき、目地の凹み、壁下部の汚れ、床の擦れ跡を含みます |
| `scripts/graphics.py` | ポスター6種、出口看板、乗換看板、誘導灯、扉の表示板、消火栓の表示を Pillow で描画します |
| `scripts/build_scene.py` | 通路・設備・照明・カメラを組み立てて Cycles でレンダリングします（出力は線形 EXR） |
| `scripts/post.py` | 発光のにじみ、レンズのベール、色収差、周辺減光をかけたあとトーンマッピングし、粒子感を加えます |

## 再現手順

```bash
pip install bpy==5.3.0 numpy scipy pillow
scripts/fetch_fonts.sh                      # Noto Sans/Serif JP など（OFL）
cd scripts
python3 build_scene.py --out ../render/corridor.exr --width 1920 --samples 400 --noise 0.008 --texres 4096
python3 post.py ../render/corridor.exr ../render/corridor.png --exposure -0.5 --view "Khronos PBR Neutral" --look None
```

GPU のない4コアの CPU で、1920×1080・400サンプルのレンダリングに約40分かかりました（実測2357秒）。
