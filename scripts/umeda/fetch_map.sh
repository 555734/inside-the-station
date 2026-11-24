#!/usr/bin/env bash
# Downloads the official Osaka Metro document that contains the Umeda Station premises map
# (避難確保・浸水防止計画 御堂筋線 梅田駅, page 9 「駅構内 避難経路図」) and extracts the page raster
# to cache/umeda/map.png. The map itself is not redistributed in this repository.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p cache/umeda
curl -sSfL -o cache/umeda/hp0116_R1umeda.pdf "https://www.osakametro.co.jp/safety/library/hinankakuhokeikaku/hp0116_R1umeda.pdf"
pdfimages -f 9 -l 9 -png cache/umeda/hp0116_R1umeda.pdf cache/umeda/page9
mv cache/umeda/page9-000.png cache/umeda/map.png
echo "cache/umeda/map.png"
