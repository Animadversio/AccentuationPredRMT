#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19205
#SBATCH --job-name=nlctrl_mass
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/%A_%a.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/logs/%A_%a.err

set -euo pipefail
MODELS=(
  AlexNet_training_seed_01
  clipag_vitb32
  dinov2_vitb14_reg
  radio_v2.5-b
  regnety_640
  resnet50
  resnet50_clip
  resnet50_dino
  resnet50_robust
  siglip2_vitb16
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/run_scale_mass.py \
  --model "$MODEL" \
  --geometry-manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --site-manifest tables/nonlinear_control/scale_manifest/site_model_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1"
