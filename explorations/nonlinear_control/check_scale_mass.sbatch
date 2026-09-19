#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=binxuwang_lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=nlctrl_check
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/check_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/check_%j.err

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/check_scale_mass.py \
  --manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1"
