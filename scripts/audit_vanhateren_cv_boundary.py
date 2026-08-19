"""Float32/float64 audit of a deterministic Van Hateren RidgeCV trial."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.validate_ffhq_disk_teacher import configure_logger, create_disk_teacher
from scripts.validate_vanhateren_disk_teacher import (
    VANHATEREN_DIR,
    stage_vanhateren_uint8,
)


def loo_path(
        X: torch.Tensor, beta: torch.Tensor, noise: torch.Tensor,
        alphas: torch.Tensor) -> torch.Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    yc = Xc @ beta + noise
    yc = yc - yc.mean()
    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval = sample_eval.clamp_min_(0)[1:]
    sample_evec = sample_evec[:, 1:]
    response_proj = sample_evec.T @ yc
    shrink = alphas[None, :] / (sample_eval[:, None] + alphas[None, :])
    residual = sample_evec @ (shrink * response_proj[:, None])
    one_minus_leverage = sample_evec.square() @ shrink
    return (residual / one_minus_leverage).square().mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=VANHATEREN_DIR)
    parser.add_argument('--scratch-dir', type=Path,
                        default=Path('/tmp/vanhateren_disk_teacher_de'))
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--trial-index', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--train-pool-size', type=int, default=2000)
    parser.add_argument('--population-size', type=int, default=20000)
    parser.add_argument('--train-patches-per-image', type=int, default=1)
    parser.add_argument('--population-patches-per-image', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=100)
    parser.add_argument('--alpha-grid-size', type=int, default=181)
    parser.add_argument('--alpha-min-exp', type=float, default=-4.0)
    parser.add_argument('--alpha-max-exp', type=float, default=5.0)
    parser.add_argument('--log-file', type=Path,
                        default=Path('/tmp/vanhateren_cv_boundary_audit.log'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The precision audit requires CUDA.')
    logger = configure_logger(args.log_file)
    total = args.train_pool_size + args.population_size
    staged_path = args.scratch_dir / (
        f'vanhateren_log_gray_{args.img_size}px_{total}_uint8.npy')
    staged = stage_vanhateren_uint8(
        args.source_dir, staged_path, args.train_pool_size,
        args.population_size, args.img_size,
        args.train_patches_per_image, args.population_patches_per_image,
        args.seed, 10, logger, no_progress=True)
    device = torch.device('cuda')
    train_pool = torch.from_numpy(np.asarray(
        staged[:args.train_pool_size], dtype=np.float32)).to(device)
    train_pool.div_(255.0)
    beta = torch.from_numpy(
        create_disk_teacher(args.img_size, 0.3)).to(device)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 1000 * args.case_index)
    for _ in range(args.trial_index):
        torch.randperm(
            args.train_pool_size, generator=generator, device=device)
        torch.randn(args.n, generator=generator, device=device)
    indices = torch.randperm(
        args.train_pool_size, generator=generator,
        device=device)[:args.n]
    noise = torch.randn(
        args.n, generator=generator, device=device) * args.sigma
    alphas_np = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)

    for dtype in (torch.float32, torch.float64):
        alphas = torch.from_numpy(alphas_np).to(device=device, dtype=dtype)
        mse = loo_path(
            train_pool[indices].to(dtype), beta.to(dtype),
            noise.to(dtype), alphas)
        selected = int(torch.argmin(mse))
        values = mse.detach().cpu().numpy()
        logger.info(
            '%s selected alpha %.9g at index %d; '
            'MSE selected %.9g, boundary %.9g, local neighbors %s',
            dtype, alphas_np[selected], selected, values[selected], values[0],
            values[max(0, selected - 2):selected + 3].tolist())


if __name__ == '__main__':
    main()
