#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=binxuwang_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=nlctrl_deepcheck
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/deep_check_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/deep_check_%j.err

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/deep_validate_scale_mass.py \
  --manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1" \
  --workers 8
