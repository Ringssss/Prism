#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/local/Prism" >&2
  exit 1
fi
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DST_DIR="$1"
rsync -a --delete "$SRC_DIR/" "$DST_DIR/"
cd "$DST_DIR"
git add -A
git commit -m "feat: add runnable Prism pack (scripts + datasets + README)" || true
echo "Synced Prism pack to repo at: $DST_DIR"
