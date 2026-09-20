#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=holygpu8a19205
#SBATCH --job-name=stein_regnet
#SBATCH --output=/tmp/stein_regnet_%A_%a.out
#SBATCH --error=/tmp/stein_regnet_%A_%a.err

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python -u explorations/nonlinear_control/run_stein_nested_pilot.py \
  --model regnety_640 \
  --geometry-manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/stein_antithetic_full_v1" \
  --directions 512 --prefixes 64 128 256 512 --images 10 \
  --geometry-start "$((2+SLURM_ARRAY_TASK_ID))" --geometry-stride 3 --max-geometries 100
