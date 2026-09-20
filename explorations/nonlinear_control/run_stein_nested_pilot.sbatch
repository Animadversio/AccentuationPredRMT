#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19205
#SBATCH --job-name=stein_nested
#SBATCH --output=/tmp/stein_nested_%A_%a.out
#SBATCH --error=/tmp/stein_nested_%A_%a.err

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
IMAGES="${STEIN_PILOT_IMAGES:-10}"
OUTPUT="${STEIN_PILOT_OUTPUT:-$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/stein_nested_v1}"
DIRECTIONS="${STEIN_PILOT_DIRECTIONS:-1024}"
if [[ -n "${STEIN_PILOT_PREFIXES:-}" ]]; then
  PREFIXES="$STEIN_PILOT_PREFIXES"
elif [[ "$DIRECTIONS" -ge 1024 ]]; then
  PREFIXES="64 128 256 512 1024"
else
  PREFIXES="64 128 256 512"
fi
GEOMETRIES="${STEIN_PILOT_GEOMETRIES:-1}"
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python -u explorations/nonlinear_control/run_stein_nested_pilot.py \
  --model "$MODEL" \
  --geometry-manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --output "$OUTPUT" \
  --directions "$DIRECTIONS" --prefixes $PREFIXES --images "$IMAGES" \
  --max-geometries "$GEOMETRIES"
