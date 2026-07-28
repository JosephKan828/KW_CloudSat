#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ====================================================
# Parse Command Line Arguments (The Switcher)
# ====================================================
# Default mode is concat
DATA_TYPE="concat"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) DATA_TYPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1. Valid options are --mode concat or --mode composite"; exit 1 ;;
    esac
    shift
done

# Validate the argument
if [[ "$DATA_TYPE" != "concat" && "$DATA_TYPE" != "composite" ]]; then
    echo "Error: Mode must be either 'concat' or 'composite'."
    exit 1
fi

echo "========================================"
echo "Starting Analysis Pipeline"
echo "Mode selected: $DATA_TYPE"
echo "========================================"

# ====================================================
# Environment Setup
# ====================================================
export MAX_CPUS=4
export OMP_NUM_THREADS=$MAX_CPUS
export MKL_NUM_THREADS=$MAX_CPUS
export OPENBLAS_NUM_THREADS=$MAX_CPUS
export VECLIB_MAXIMUM_THREADS=$MAX_CPUS
export NUMEXPR_NUM_THREADS=$MAX_CPUS

# ====================================================
# Execution
# ====================================================
echo "1. Running Linear Relations..."
python QR_w_Relation.py --data_type $DATA_TYPE

echo "2. Running Rad Mode Predictions..."
python Rad_mode_predict.py --data_type $DATA_TYPE

echo "3. Running Rad Mode Verifications..."
python Rad_mode_verification.py --data_type $DATA_TYPE

echo "Analysis complete!"
