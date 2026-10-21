#!/usr/bin/env bash
set -euo pipefail

# The upstream Parquet has a published LFS SHA-256. A mirror is used only as a
# transport because AutoDL cannot currently resolve huggingface.co reliably.
revision=65877096c24ffa7abc4e4fa5edb95cf3413a5674
expected_sha=534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target_dir="$project_dir/data/raw/dapo_math_17k"
target="$target_dir/dapo-math-17k.parquet"
partial="$target.part"
url="https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/$revision/data/dapo-math-17k.parquet"

mkdir -p "$target_dir"
if [[ ! -f "$target" ]]; then
    curl --fail --location --retry 8 --retry-delay 5 --retry-all-errors \
        --connect-timeout 15 --continue-at - --output "$partial" "$url"
    printf '%s  %s\n' "$expected_sha" "$partial" | sha256sum --check --status
    mv "$partial" "$target"
fi
printf '%s  %s\n' "$expected_sha" "$target" | sha256sum --check
printf 'repo=BytedTsinghua-SIA/DAPO-Math-17k\nrevision=%s\nsha256=%s\nsource=%s\n' \
    "$revision" "$expected_sha" "$url" > "$target_dir/source.txt"
stat -c 'downloaded %n (%s bytes)' "$target"
