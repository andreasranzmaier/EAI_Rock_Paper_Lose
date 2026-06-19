#!/usr/bin/env bash
# Pulls the Pi's OpenCV 4.10 libs + headers into third_party/pi-sysroot/
# so the cross-compile can link against the exact version installed on the
# target. Must run from a host that can reach the Pi (e.g. your Mac, not the
# devcontainer).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

: "${PI_USER:?Set PI_USER in .env}"
: "${PI_HOST:?Set PI_HOST in .env}"

SYSROOT="third_party/pi-sysroot"
mkdir -p "$SYSROOT/usr/lib/aarch64-linux-gnu" "$SYSROOT/usr/include"

echo "Syncing OpenCV libs from $PI_USER@$PI_HOST..."
rsync -avzL \
  --include='libopencv_core.so*' \
  --include='libopencv_imgproc.so*' \
  --include='*/' --exclude='*' \
  "$PI_USER@$PI_HOST:/usr/lib/aarch64-linux-gnu/" \
  "$SYSROOT/usr/lib/aarch64-linux-gnu/"

echo "Syncing OpenCV headers..."
rsync -avz \
  "$PI_USER@$PI_HOST:/usr/include/opencv4/" \
  "$SYSROOT/usr/include/opencv4/"

echo "Done. Sysroot at $SYSROOT"
ls -lh "$SYSROOT/usr/lib/aarch64-linux-gnu/" | head
