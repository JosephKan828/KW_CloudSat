#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ====================================================
# Environment Setup
# ====================================================
export MAX_CPUS=4
export OMP_NUM_THREADS=$MAX_CPUS
export MKL_NUM_THREADS=$MAX_CPUS
export OPENBLAS_NUM_THREADS=$MAX_CPUS
export VECLIB_MAXIMUM_THREADS=$MAX_CPUS
export NUMEXPR_NUM_THREADS=$MAX_CPUS

echo "Starting Preprocessing Pipeline..."

echo "========================================"
echo "1. Running KW Selection (Iterating internally over all bands)..."
python KW_selection.py

echo "========================================"
echo "2. Running ERA5 Composite (Iterating internally over all bands)..."
python ERA5_composite.py

echo "========================================"
echo "3. Running QR Composite (Iterating internally over all bands)..."
python QR_composite.py

echo "========================================"
echo "Preprocessing complete! This data is now ready for analysis."
