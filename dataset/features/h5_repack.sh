#!/usr/bin/env bash
set -euo pipefail

dir="/home/ai/Projects/Mahsa/datasets/grid/"

for split in test; do
  echo "Scanning: $dir (pattern: grid_${split}_features_chunk*.h5)"
  shopt -s nullglob
  for f in "$dir"/grid_${split}_features_chunk*.h5; do
    tmp="${f%.h5}.repacked.h5"
    bak="${f}.bak"

    echo "Repacking: $f"
    if ! h5repack "$f" "$tmp"; then
      echo "FAILED during h5repack: $f"
      rm -f "$tmp"
      continue
    fi

    if ! h5ls -r "$tmp" >/dev/null 2>&1; then
      echo "Verification failed: $f"
      rm -f "$tmp"
      continue
    fi

    cp -a --reflink=auto "$f" "$bak" || cp -a "$f" "$bak"
    mv -f "$tmp" "$f"   # atomic replace on same filesystem
    rm -f "$bak"
    echo "✔ Done: $f"
  done
done
