#!/bin/bash
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=nlctrl_biology
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/biological_validation_v1/logs/%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/biological_validation_v1/logs/%j.err
set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
: "${STORE_DIR:?STORE_DIR must be configured}"
# Source HDF5s are immutable read-only inputs. Network filesystem locking stalls.
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
"$PYTHON" -u explorations/nonlinear_control/validate_biology.py preprocess
"$PYTHON" -u explorations/nonlinear_control/validate_biology.py analyze
