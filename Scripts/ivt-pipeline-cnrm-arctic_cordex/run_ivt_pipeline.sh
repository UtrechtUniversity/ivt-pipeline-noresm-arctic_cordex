#!/bin/bash
set -eo pipefail
#SBATCH --job-name=ivt_CNRM_CORDEX
#SBATCH --output=/ec/res4/scratch/nld1254/cnrm/ivt-pipeline/ivt_CNRM_CORDEX_%j.log
#SBATCH --error=/ec/res4/scratch/nld1254/cnrm/ivt-pipeline/ivt_CNRM_CORDEX_%j.log
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.gavras-vangarderen@uu.nl

# --------------------------------------------------------
# IVT Processing Pipeline Wrapper Script
# --------------------------------------------------------
# This script automates the workflow for calculating
# Integrated Vapor Transport (IVT) from the GCM CNRM model of NorESM.
# It uses model layers 300-1000hPa (9 layers: 300 400 500 600 700 750 850 925 1000).
#
# The script relies on a specific Python version:
# Default path: /usr/local/apps/python3/3.12.9-01/bin/python3
# Modify the PYTHON variable below if your environment differs.
#
# Workflow steps executed sequentially:
#   1) Extract pressure levels and select Arctic region
#   2) Calculate IVT
#   3) Regrid to CORDEX Arctic grid and apply polar point wrapping
#
# Users specify the climate model (GCM), start year, end year,
# and optionally start and end months for partial-year processing.
# If no arguments are provided, the script defaults to:
#   GCM = "CNRMESM21"
#   START_YEAR = 1985
#   END_YEAR = 1985
#   START_MONTH = 02
#   END_MONTH = 02
#
# The script automatically selects the appropriate experiment
# (historical or ssp370) based on the year and sets the
# corresponding model member ID.
#
# Paths to input data, intermediate files, and final outputs
# are defined in the script but can be modified as needed.
#
# ----------------- Usage Options -------------------------
#
# 1) Run interactively or from command line:
#
#    ./run_ivt_pipeline.sh [GCM] [START_YEAR] [END_YEAR] [START_MONTH] [END_MONTH]
#
#    Examples:
#      ./run_ivt_pipeline.sh
#        # Runs default: CNRMESM21 for Feb 1985
#
#      ./run_ivt_pipeline.sh NORESM2MM 2015 2015 03 06
#        # Runs NorESM2-MM for Mar-Jun 2015
#
#      ./run_ivt_pipeline.sh CNRMESM21 2014 2014 12 12
#        # Runs CNRM-ESM2.1 for Dec 2014 only
#
# 2) Submit as a SLURM batch job:
#
#    sbatch run_ivt_pipeline.sh [GCM] [START_YEAR] [END_YEAR] [START_MONTH] [END_MONTH]
#
#    Ensure SLURM header directives (e.g. #SBATCH) are present at the top of this script.
#
#    SLURM parameters such as CPUs, memory, and time should be adjusted
#    in the header according to your cluster and job needs.
#
#    Example:
#      sbatch run_ivt_pipeline.sh CNRMESM21 1985 1985 02 02
#
# --------------------------------------------------------
# Note:
# Older output files with the same name will be overwritten.
# --------------------------------------------------------


module load cdo
module load nco
# for proper NCO usage in pipeline:
module load hdf5/1.14.6   
export HDF5_USE_FILE_LOCKING=FALSE

GCM=${1:-"CNRMESM21"}
START_YEAR=${2:-1985}
END_YEAR=${3:-1985}
START_MONTH=${4:-01}
END_MONTH=${5:-12}

# === Paths ===
BASE_DIR="/ec/res4/scratch/nld1254/cnrm"
MODEL_INPUT="${BASE_DIR}/6hourly" 
TRACKABLE_OUTPUT="${BASE_DIR}/cordex"
IVT_OUTPUT="${BASE_DIR}/ivt"
ARCHIVE="ec:/nld1254/IVT/CNRM"

mkdir -p "$MODEL_INPUT" "$TRACKABLE_OUTPUT" "$IVT_OUTPUT"

PYTHON='/usr/local/apps/python3/3.12.9-01/bin/python3'

if ! [[ "$START_MONTH" =~ ^(0[1-9]|1[0-2])$ ]] || ! [[ "$END_MONTH" =~ ^(0[1-9]|1[0-2])$ ]]; then
  echo "Error: START_MONTH and END_MONTH must be two-digit months between 01 and 12"
  exit 1
fi

if [[ $GCM == "CNRMESM21" ]]; then
  MEMBER="r1i1p1f2"
else
  echo "Unsupported GCM: $GCM"
  exit 1
fi

MONTH_LIST=()
for YEAR in $(seq $START_YEAR $END_YEAR); do
  if [[ "$YEAR" -eq "$START_YEAR" && "$YEAR" -eq "$END_YEAR" ]]; then
    MONTH_START=$START_MONTH
    MONTH_END=$END_MONTH
  elif [[ "$YEAR" -eq "$START_YEAR" ]]; then
    MONTH_START=$START_MONTH
    MONTH_END=12
  elif [[ "$YEAR" -eq "$END_YEAR" ]]; then
    MONTH_START=01
    MONTH_END=$END_MONTH
  else
    MONTH_START=01
    MONTH_END=12
  fi
  for MONTH in $(seq -w $MONTH_START $MONTH_END); do
    MONTH_LIST+=("${YEAR}${MONTH}")
  done
done

echo "Running IVT for: $GCM | $MEMBER"
for m in "${MONTH_LIST[@]}"; do
  echo "  - $m"
done

#read -p "🔁 Continue? (yes/y): " confirm
#[[ "$confirm" != "yes" && "$confirm" != "y" ]] && exit 0

for YYYYMM in "${MONTH_LIST[@]}"; do
  YEAR=${YYYYMM:0:4}
  MONTH=${YYYYMM:4:2}
  EXP=$([[ $YEAR -lt 2015 ]] && echo "historical" || echo "ssp370")


  echo ">>> Step 1: Download & prepare data for $YEAR-$MONTH ..."
  $PYTHON ./Download_process_cmip6.py \
    --gcm "$GCM" \
    --year "$YEAR" \
    --month "$MONTH" \
    --output_dir "$MODEL_INPUT"  # output to the same dir

  echo ">>> Step 2: Calculate IVT for $YEAR-$MONTH ..."
  ./calculate_ivt.sh "$YYYYMM" "$BASE_DIR" "$MODEL_INPUT" "$IVT_OUTPUT"

  echo ">>> Step 3: wrap and add polar point, regrid and cut to CORDEX Arctic for $YEAR-$MONTH ..."
  $PYTHON Regrid_RotPolar_CORDEX.py "$YEAR" "$MONTH" "$IVT_OUTPUT" "$TRACKABLE_OUTPUT" "$BASE_DIR"

 echo ">>> Step 4: Archive to tape for $YEAR-$MONTH ..."
 ./archive_IVT_cnrmCORDEX.sh "$YEAR" "$MONTH" "$TRACKABLE_OUTPUT" "$ARCHIVE"
  echo ">>> Done processing $YEAR-$MONTH"
  
done

echo "✅ All processing complete."
