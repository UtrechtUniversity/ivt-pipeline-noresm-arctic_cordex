#!/bin/bash
set -e

# Usage: $0 <YEAR> <MONTH> <input_dir> <archive_dir>
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <YEAR> <MONTH> <input_dir> <archive_dir>"
  exit 1
fi

YEAR=$1
MONTH=$2
INPUT_DIR=$3
ARCHIVE_DIR="${4%/}"   # strip any trailing slash

filenm="IVT_NORESM_CORDEX_${YEAR}${MONTH}.nc"
src="$INPUT_DIR/$filenm"
dst="$ARCHIVE_DIR/$filenm"

echo "Archiving $filenm to $dst …"
if [ ! -f "$src" ]; then
  echo "❌ Source file not found: $src"
  exit 2
fi

ecp "$src" "$dst"
echo "✅ Archive successful: $dst"
