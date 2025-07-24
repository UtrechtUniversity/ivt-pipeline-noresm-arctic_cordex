#!/bin/bash
set -e
# archive IVT file to tape archive


# === USAGE CHECK ===
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <YEAR> <MONTH> <input_dir> <output_dir>"
  exit 1
fi

# === INPUTS FROM WRAPPER ===
YEAR=$1      
MONTH=$2 
INPUT_DIR=$3   # CNRM IVT in polar cordex grid
ARCHIVE=$4  # path on tape archive

filenm=IVT_CNRM_CORDEX_$YEAR$MONTH.nc

echo "Archiving $filenm to $ARCHIVE ..."
ecp "$INPUT_DIR/$filenm" "$ARCHIVE"

echo "Archive successful"

