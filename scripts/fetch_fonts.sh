#!/usr/bin/env bash
# Fetches the OFL-licensed Google Fonts used for posters and signs into ./fonts
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p fonts
tmp=$(mktemp -d)
for p in noto-sans-jp noto-serif-jp zen-maru-gothic dela-gothic-one m-plus-1p; do
  (cd "$tmp" && npm pack "@expo-google-fonts/$p" >/dev/null)
done
for f in "$tmp"/*.tgz; do tar -xzf "$f" -C "$tmp"; done
find "$tmp" -name "*.ttf" -exec cp {} fonts/ \;
rm -rf "$tmp"
ls fonts
