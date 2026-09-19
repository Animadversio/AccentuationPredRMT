#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=1-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19205
#SBATCH --job-name=nlctrl_regnet
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/regnet_%A_%a.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/regnet_%A_%a.err

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/run_scale_mass.py \
  --model regnety_640 \
  --geometry-index "$SLURM_ARRAY_TASK_ID" \
  --geometry-manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --site-manifest tables/nonlinear_control/scale_manifest/site_model_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1"
