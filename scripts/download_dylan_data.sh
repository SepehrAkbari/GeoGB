#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/external-drive/GeoGB-data" >&2
  exit 1
fi

target_dir="$1"
archive="$target_dir/deepgroebner-data.tar.gz"
url="https://zenodo.org/records/4724144/files/deepgroebner-data.tar.gz?download=1"
expected_md5="9cf4d05d5b71825f947befd4846feeb5"

mkdir -p "$target_dir"
curl --fail --location --continue-at - --output "$archive" "$url"

actual_md5="$(md5 -q "$archive")"
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "Checksum mismatch: expected $expected_md5, got $actual_md5" >&2
  exit 1
fi

echo "Download verified: $archive"
echo "The archive is 17.2 GB. Extract it on the external drive, then follow docs/DATA.md."
