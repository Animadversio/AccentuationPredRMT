#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=holygpu8a19205
#SBATCH --job-name=nlctrl_smoke
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_smoke/logs/%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_smoke/logs/%j.err

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/run_scale_mass.py \
  --model AlexNet_training_seed_01 \
  --geometry-manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --site-manifest tables/nonlinear_control/scale_manifest/site_model_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_smoke" \
  --directions 16 --centers 2 --images 1 --max-geometries 1
