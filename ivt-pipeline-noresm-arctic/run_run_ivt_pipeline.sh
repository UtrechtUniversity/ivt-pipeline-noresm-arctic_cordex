#!/bin/bash
# --------------------------------------------------------
# Batch Submission Wrapper for IVT Processing Pipeline
# --------------------------------------------------------
# This script automates running the main IVT pipeline (`run_ivt_pipeline.sh`)
# in sequential SLURM jobs, splitting a large year range into smaller chunks of 3 years.
#
# It handles SLURM job dependencies to ensure jobs run one after another.
# Users specify:
#   - Climate model (GCM)
#   - Start year and end year for processing
#   - Optional start and end months for partial-year runs
#
# The script submits multiple SLURM jobs, each calling the main pipeline with
# appropriate arguments, respecting HPC time limits (e.g., 8 hours).
#
# Usage:
#   ./run_run_ivt_pipeline.sh <GCM> <START_YEAR> <END_YEAR> [START_MONTH] [END_MONTH]
#
# Example:
#   ./run_run_ivt_pipeline.sh NORESM2MM 1985 1990 01 12 
#   # Runs the IVT pipeline for each year 1985 through 1990 in separate jobs
#
# This wrapper improves HPC efficiency by chunking the workload and
# managing job dependencies automatically. 
#
# NOTE: The ESGF server hosting scenario data is not stable under parallel access. 
# To avoid crashes and failed downloads, jobs are run sequentially using SLURM dependencies. 
# Each chunk waits for the previous one to complete successfully before starting.
# --------------------------------------------------------

# Parameters (input)
START_YEAR=$1
END_YEAR=$2
START_MONTH=${3:-01}
END_MONTH=${4:-12}
YEARS_PER_JOB=3  # years per batch

PREV_JOB_ID=""

current_start=$START_YEAR
while [ "$current_start" -le "$END_YEAR" ]; do
  current_end=$((current_start + YEARS_PER_JOB - 1))
  if [ "$current_end" -gt "$END_YEAR" ]; then
    current_end=$END_YEAR
  fi

  # Only pass months if this is the first or last year chunk, else default to full year
  if [ "$current_start" -eq "$START_YEAR" ] && [ "$current_end" -eq "$END_YEAR" ]; then
    # Single chunk with possible months
    months_args="$START_MONTH $END_MONTH"
  elif [ "$current_start" -eq "$START_YEAR" ]; then
    months_args="$START_MONTH 12"
  elif [ "$current_end" -eq "$END_YEAR" ]; then
    months_args="01 $END_MONTH"
  else
    months_args="01 12"
  fi

  if [ -z "$PREV_JOB_ID" ]; then
    jobid=$(sbatch --parsable run_ivt_pipeline.sh "$current_start" "$current_end" $months_args)
  else
    jobid=$(sbatch --parsable --dependency=afterok:$PREV_JOB_ID run_ivt_pipeline.sh "$current_start" "$current_end" $months_args)
  fi
  echo "Submitted job $jobid for years $current_start-$current_end"

  PREV_JOB_ID=$jobid
  current_start=$((current_end + 1))
done
