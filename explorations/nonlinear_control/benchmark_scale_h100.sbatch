#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=nlctrl_bench
#SBATCH --output=/tmp/nonlinear_control_scale_benchmark_%j.log

set -euo pipefail
cd /n/home12/binxuwang/Github/AccentuationPredRMT
source /n/home12/binxuwang/.bashrc
mamba activate torch2
python explorations/nonlinear_control/benchmark_scale_models.py \
  --manifest tables/nonlinear_control/scale_manifest/geometry_manifest.csv \
  --output tables/nonlinear_control/scale_manifest/h100_benchmark.json \
  --repeats 3
