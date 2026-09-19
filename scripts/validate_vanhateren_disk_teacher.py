"""Paired fixed-ridge and RidgeCV experiment on Van Hateren image patches.

The setup matches the FFHQ disk-teacher comparison: 100 x 100 grayscale
inputs, a binary disk teacher, n=1000, dense exact-LOOCV RidgeCV, and a fixed
alpha=100 (lambda=0.1).  Source images are split before patch extraction so
the training pool and population covariance use disjoint photographs.

Plot-ready summaries are saved separately from per-noise trial caches, making
``--plot-only`` independent of the image data and GPU.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import numpy as np
import torch

from scripts.compare_ffhq_fixed_vs_cv import (
    add_theory_order_columns,
    plot_policy_gap_comparison,
)
from scripts.validate_ffhq_disk_teacher import (
    aggregate,
    configure_logger,
    create_disk_teacher,
    fit_metrics,
    select_de_alpha,
    theory_metrics,
)
from scripts.validate_r2_peer_review import float_tag
from scripts.storage_paths import configured_bulk_path, require_bulk_path


REPO_ROOT = Path(__file__).resolve().parents[1]
VANHATEREN_DIR = Path(
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/'
    'Datasets/vanhateren_natural_stimuli')
SUMMARY_PATH = REPO_ROOT / 'tables' / 'vanhateren_fixed_vs_cv_summary.csv'
SPECTRUM_PATH = REPO_ROOT / 'tables' / 'vanhateren_disk_teacher_spectrum.npz'
CASE_DIR = configured_bulk_path('tables/vanhateren_fixed_vs_cv_cases')
FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'model_selection' /
    'vanhateren_fixed_vs_cv_gen_acc_gap.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'vanhateren_fixed_vs_cv.log'

# Exactly the 26 normalized noise levels used in the FFHQ comparison.  Using
# sigma^2 / S makes the two datasets comparable despite different signal power.
DEFAULT_NOISE_RATIOS = np.asarray([
    0.0,
    1.47529557057e-08,
    1.32776601351e-07,
    1.47529557057e-06,
    1.32776601351e-05,
    3.68823892642e-05,
    7.22894829579e-05,
    1.47529557057e-04,
    3.31941503378e-04,
    5.90118228228e-04,
    1.32776601351e-03,
    3.68823892642e-03,
    1.0e-02,
    1.47529557057e-02,
    3.0e-02,
    5.90118228228e-02,
    1.0e-01,
    1.32776601351e-01,
    2.36047291291e-01,
    3.0e-01,
    5.31106405405e-01,
    1.0,
    1.47529557057,
    3.0,
    4.77995764864,
    10.0,
])


def load_vanhateren_image(path: Path) -> np.ndarray:
    """Read one big-endian 1024 x 1536 Van Hateren luminance image."""
    image = np.fromfile(path, dtype='>u2')
    if image.size != 1024 * 1536:
        raise ValueError(f'Unexpected Van Hateren image size at {path}')
    return image.reshape(1024, 1536)


def stage_vanhateren_uint8(
        source_dir: Path, cache_path: Path, train_pool_size: int,
        population_size: int, img_size: int,
        train_patches_per_image: int, population_patches_per_image: int,
        seed: int, probe_images: int, logger,
        no_progress: bool = False) -> np.memmap:
    """Extract deterministic log-luminance patches into a local uint8 cache."""
    total_patches = train_pool_size + population_size
    expected_shape = (total_patches, img_size * img_size)
    meta_path = cache_path.with_suffix('.meta.npz')
    expected_config = {
        'shape': np.asarray(expected_shape),
        'train_pool_size': train_pool_size,
        'population_size': population_size,
        'img_size': img_size,
        'train_patches_per_image': train_patches_per_image,
        'population_patches_per_image': population_patches_per_image,
        'seed': seed,
    }
    if cache_path.exists() and meta_path.exists():
        with np.load(meta_path, allow_pickle=False) as metadata:
            matches = all(
                key in metadata and np.array_equal(metadata[key], value)
                for key, value in expected_config.items())
        if matches:
            logger.info(
                'Loading local Van Hateren patch cache %s, shape=%s',
                cache_path, expected_shape)
            return np.load(cache_path, mmap_mode='r')

    files = sorted(source_dir.glob('*.iml'))
    if not files:
        raise FileNotFoundError(f'No .iml files found in {source_dir}')
    train_images = int(np.ceil(
        train_pool_size / train_patches_per_image))
    population_images = int(np.ceil(
        population_size / population_patches_per_image))
    total_images = train_images + population_images
    if total_images > len(files):
        raise ValueError(
            f'Need {total_images} source images but found only {len(files)}')

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(files), size=total_images, replace=False)
    chosen_files = [files[index] for index in chosen_indices]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    patches = np.lib.format.open_memmap(
        cache_path, mode='w+', dtype=np.uint8, shape=expected_shape)
    source_index = np.empty(total_patches, dtype=np.int32)
    crop_row = np.empty(total_patches, dtype=np.int16)
    crop_col = np.empty(total_patches, dtype=np.int16)

    logger.info(
        'Staging %d patches from %d disjoint source images into %s',
        total_patches, total_images, cache_path)
    logger.info(
        'Split: %d training images -> %d patches; '
        '%d population images -> %d patches',
        train_images, train_pool_size, population_images, population_size)
    start = time.perf_counter()
    try:
        from tqdm.auto import tqdm
        iterator = enumerate(chosen_files)
        if not no_progress:
            iterator = tqdm(
                iterator, total=total_images,
                desc='stage Van Hateren patches', unit='image')
    except ImportError:
        iterator = enumerate(chosen_files)

    patch_index = 0
    for image_index, path in iterator:
        image = load_vanhateren_image(path).astype(np.float32)
        # Match the FFHQ [0, 1] pixel scale while retaining the conventional
        # log-luminance transform used for Van Hateren images.
        image = np.log1p(image) * (255.0 / np.log(65536.0))
        group_stop = (
            train_pool_size if image_index < train_images else total_patches)
        group_start = 0 if image_index < train_images else train_pool_size
        if patch_index < group_start:
            patch_index = group_start
        patches_per_image = (
            train_patches_per_image if image_index < train_images
            else population_patches_per_image)
        count = min(patches_per_image, group_stop - patch_index)
        for _ in range(max(count, 0)):
            row = int(rng.integers(0, image.shape[0] - img_size + 1))
            col = int(rng.integers(0, image.shape[1] - img_size + 1))
            patch = np.rint(image[
                row:row + img_size, col:col + img_size]).astype(np.uint8)
            patches[patch_index] = patch.ravel()
            source_index[patch_index] = image_index
            crop_row[patch_index] = row
            crop_col[patch_index] = col
            patch_index += 1
        if image_index + 1 == min(probe_images, total_images):
            elapsed = time.perf_counter() - start
            projected = elapsed * total_images / (image_index + 1)
            logger.info(
                'Staging probe: %d images in %.2fs; projected %.1fs '
                '(%.1fmin) for %d images',
                image_index + 1, elapsed, projected, projected / 60,
                total_images)

    if patch_index != total_patches:
        raise RuntimeError(
            f'Staged {patch_index} patches, expected {total_patches}')
    patches.flush()
    elapsed = time.perf_counter() - start
    metadata = dict(expected_config)
    metadata.update({
        'elapsed_seconds': elapsed,
        'source_files': np.asarray([path.name for path in chosen_files]),
        'source_index': source_index,
        'crop_row': crop_row,
        'crop_col': crop_col,
        'transform': np.asarray('round(255*log1p(raw)/log(65536))'),
    })
    np.savez_compressed(meta_path, **metadata)
    logger.info(
        'Staged Van Hateren patches in %.1fs (%.1f patches/s)',
        elapsed, total_patches / elapsed)
    return np.load(cache_path, mmap_mode='r')


def compute_population_spectrum(
        staged: np.ndarray, train_pool_size: int, population_size: int,
        beta: np.ndarray, cache_dir: Path, device: torch.device,
        logger, force: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Compute and cache the independent patch-population eigensystem."""
    d = len(beta)
    tag = f'vh_train{train_pool_size}_pop{population_size}_d{d}'
    eigvec_path = cache_dir / f'{tag}_eigvec.npy'
    compact_path = cache_dir / f'{tag}_spectrum.npz'
    if force:
        for path in (eigvec_path, compact_path):
            if path.exists():
                path.unlink()
    if eigvec_path.exists() and compact_path.exists():
        logger.info('Loading cached Van Hateren spectrum from %s', compact_path)
        with np.load(compact_path) as data:
            eigenvalues = np.asarray(data['eigenvalues'])
            beta_proj = np.asarray(data['beta_proj'])
        eigvec = torch.tensor(np.asarray(
            np.load(eigvec_path, mmap_mode='r')), device=device)
        return (
            torch.from_numpy(eigenvalues).to(device), eigvec,
            eigenvalues, beta_proj)

    start = train_pool_size
    stop = start + population_size
    load_start = time.perf_counter()
    population = torch.from_numpy(
        np.asarray(staged[start:stop], dtype=np.float32)).to(device)
    population.div_(255.0)
    population.sub_(population.mean(dim=0, keepdim=True))
    logger.info(
        'Population tensor ready in %.1fs; computing %d x %d covariance',
        time.perf_counter() - load_start, d, d)
    covariance_start = time.perf_counter()
    covariance = population.T @ population / population_size
    del population
    torch.cuda.empty_cache()
    logger.info('Covariance completed in %.1fs; starting eigh',
                time.perf_counter() - covariance_start)
    eigh_start = time.perf_counter()
    eigval, eigvec = torch.linalg.eigh(covariance)
    del covariance
    eigval = torch.flip(eigval, dims=[0]).clamp_min_(0)
    eigvec = torch.flip(eigvec, dims=[1]).contiguous()
    beta_t = torch.from_numpy(beta).to(device)
    beta_proj_t = eigvec.T @ beta_t
    logger.info('Population eigh completed in %.1fs',
                time.perf_counter() - eigh_start)

    cache_dir.mkdir(parents=True, exist_ok=True)
    eigenvalues = eigval.cpu().numpy()
    beta_proj = beta_proj_t.cpu().numpy()
    np.save(eigvec_path, eigvec.cpu().numpy())
    np.savez_compressed(
        compact_path, eigenvalues=eigenvalues, beta_proj=beta_proj)
    SPECTRUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SPECTRUM_PATH, eigenvalues=eigenvalues, beta_proj=beta_proj,
        train_pool_size=train_pool_size, population_size=population_size,
        image_size=int(np.sqrt(d)), transform=np.asarray(
            'round(255*log1p(raw)/log(65536))/255'))
    logger.info('Cached eigensystem at %s and %s',
                eigvec_path, compact_path)
    return eigval, eigvec, eigenvalues, beta_proj


def ridge_cv_and_fixed_fit(
        X: torch.Tensor, beta: torch.Tensor, sigma: float,
        alphas: torch.Tensor, fixed_alpha: float,
        generator: torch.Generator
        ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit paired CV and fixed ridge using one noise draw/eigendecomposition.

    Exact LOOCV is evaluated in float64.  In the nearly interpolating regime,
    float32 can corrupt the ratio of two alpha-scaled quantities and
    spuriously select the smallest alpha even with the cancellation-free
    residual formula.
    """
    n = X.shape[0]
    Xc = X - X.mean(dim=0, keepdim=True)
    noise = torch.randn(n, device=X.device, generator=generator) * sigma
    yc = Xc @ beta + noise
    yc = yc - yc.mean()
    output_dtype = X.dtype
    Xc = Xc.to(torch.float64)
    yc = yc.to(torch.float64)
    alphas = alphas.to(torch.float64)
    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval = sample_eval.clamp_min_(0)
    response_proj = sample_evec.T @ yc
    positive_eval = sample_eval[1:]
    positive_evec = sample_evec[:, 1:]
    positive_response = response_proj[1:]

    residual_shrink = alphas[None, :] / (
        positive_eval[:, None] + alphas[None, :])
    residual_path = positive_evec @ (
        residual_shrink * positive_response[:, None])
    one_minus_leverage = positive_evec.square() @ residual_shrink
    loo_mse = (residual_path / one_minus_leverage).square().mean(dim=0)
    selected_index = int(torch.argmin(loo_mse))
    selected_alpha = alphas[selected_index]
    cv_dual = positive_evec @ (
        positive_response / (positive_eval + selected_alpha))
    fixed_dual = positive_evec @ (
        positive_response / (positive_eval + fixed_alpha))
    return (
        (Xc.T @ cv_dual).to(output_dtype),
        (Xc.T @ fixed_dual).to(output_dtype),
        float(selected_alpha),
    )


def case_path(args: argparse.Namespace, sigma: float) -> Path:
    case_dir = require_bulk_path(
        CASE_DIR, 'Van Hateren fixed-vs-CV case caches')
    return case_dir / (
        f'vanhateren_disk_d{args.img_size ** 2}_n{args.n}'
        f'_sigma{float_tag(sigma)}_trials{args.n_trials}'
        f'_pop{args.population_size}_seed{args.seed}'
        f'_alpha{args.alpha_grid_size}_emin{float_tag(args.alpha_min_exp)}'
        f'_emax{float_tag(args.alpha_max_exp)}'
        f'_fixed{float_tag(args.fixed_alpha)}.npz')


def save_case(
        path: Path, rows: dict[str, dict[str, object]],
        arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for policy, row in rows.items():
        payload.update({
            f'{policy}_row_{key}': np.asarray(value)
            for key, value in row.items()
        })
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def load_case(path: Path) -> tuple[dict[str, dict[str, object]],
                                   dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        rows: dict[str, dict[str, object]] = {'cv': {}, 'fixed': {}}
        arrays: dict[str, np.ndarray] = {}
        for key in data.files:
            matched = False
            for policy in rows:
                prefix = f'{policy}_row_'
                if key.startswith(prefix):
                    value = data[key]
                    rows[policy][key.removeprefix(prefix)] = (
                        value.item() if value.ndim == 0 else value.tolist())
                    matched = True
                    break
            if not matched:
                arrays[key] = np.asarray(data[key])
    return rows, arrays


def summarize_policy(
        policy: str, sigma: float, signal_power: float,
        alpha: float, n: int, theory: dict[str, float],
        trials: dict[str, np.ndarray], case: Path,
        elapsed: float, args: argparse.Namespace
        ) -> dict[str, object]:
    row: dict[str, object] = {
        'dataset': 'vanhateren',
        'policy': policy,
        'sigma': sigma,
        'noise_signal_ratio': sigma ** 2 / signal_power,
        'signal_power': signal_power,
        'alpha': alpha,
        'lambda': alpha / n,
        'n': n,
        'd': args.img_size ** 2,
        'n_trials': args.n_trials,
        'train_pool_size': args.train_pool_size,
        'population_size': args.population_size,
        'train_patches_per_image': args.train_patches_per_image,
        'population_patches_per_image': args.population_patches_per_image,
        'seed': args.seed,
        'case_path': str(case),
        'mc_elapsed_seconds': elapsed,
        'theory_gen_error_normalized': (
            float(theory['theory_gen_error']) / signal_power),
        'theory_acc_error_normalized': (
            float(theory['theory_acc_error']) / signal_power),
        'theory_acc_error_ratio_normalized': (
            float(theory['theory_acc_error_ratio_of_expectations'])
            / signal_power),
        'theory_acc_error_second_order_normalized': (
            float(theory['theory_acc_error_second_order']) / signal_power),
        'theory_r2_gen': float(theory['theory_r2_gen']),
        'theory_r2_acc': float(theory['theory_r2_acc']),
        'theory_r2_acc_ratio': float(
            theory['theory_r2_acc_ratio_of_expectations']),
        'theory_r2_acc_second_order': float(
            theory['theory_r2_acc_second_order']),
        'theory_acc_ratio_leading_mean': float(
            theory['theory_acc_alignment']),
        'theory_acc_ratio_corrected_mean': float(
            theory['theory_acc_ratio_corrected_mean']),
        'theory_acc_ratio_mean_correction': float(
            theory['theory_acc_ratio_mean_correction']),
        'theory_acc_ratio_variance': float(
            theory['theory_acc_ratio_variance']),
        'theory_r2_acc_second_order_correction': float(
            theory['theory_r2_acc_second_order_correction']),
        'theory_weight_error': float(theory['theory_weight_error']),
    }
    for metric, values in trials.items():
        mean, std, se = aggregate(values)
        row[f'mc_{metric}'] = mean
        row[f'mc_{metric}_std'] = std
        row[f'mc_{metric}_se'] = se
        row[f'mc_{metric}_median'] = float(np.median(values))
        row[f'mc_{metric}_q25'] = float(np.quantile(values, 0.25))
        row[f'mc_{metric}_q75'] = float(np.quantile(values, 0.75))
    for metric in ('gen_error', 'acc_error'):
        row[f'mc_{metric}_normalized'] = (
            float(row[f'mc_{metric}']) / signal_power)
        row[f'mc_{metric}_normalized_se'] = (
            float(row[f'mc_{metric}_se']) / signal_power)
    return row


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with SUMMARY_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.chmod(0o644)


def read_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUMMARY_PATH.open(newline='') as handle:
        for source in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in source.items():
                if value in ('', None):
                    row[key] = np.nan
                elif key in ('dataset', 'policy', 'case_path'):
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def render(rows: list[dict[str, object]], fixed_alpha: float, n: int) -> None:
    signal_power = float(rows[0]['signal_power'])
    plot_policy_gap_comparison(
        rows, fixed_alpha, signal_power, n,
        dataset_title=r'Van Hateren $100\times100$ disk teacher',
        figure_path=FIGURE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=VANHATEREN_DIR)
    parser.add_argument('--scratch-dir', type=Path,
                        default=Path('/tmp/vanhateren_disk_teacher_de'))
    parser.add_argument('--img-size', type=int, default=100)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--train-pool-size', type=int, default=2000)
    parser.add_argument('--population-size', type=int, default=20000)
    parser.add_argument('--train-patches-per-image', type=int, default=1)
    parser.add_argument('--population-patches-per-image', type=int, default=10)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--fixed-alpha', type=float, default=100.0)
    parser.add_argument('--alpha-grid-size', type=int, default=181)
    parser.add_argument('--alpha-min-exp', type=float, default=-4.0)
    parser.add_argument('--alpha-max-exp', type=float, default=5.0)
    parser.add_argument('--noise-ratios', type=float, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--benchmark-trials', type=int, default=3)
    parser.add_argument('--staging-probe-images', type=int, default=10)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--stage-only', action='store_true',
                        help='Stage/cache patches and exit before GPU work.')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--force-spectrum', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    if args.plot_only:
        rows = read_summary()
        fixed = sorted({
            float(row['alpha']) for row in rows if row['policy'] == 'fixed'})
        if len(fixed) != 1:
            raise RuntimeError(f'Expected one fixed alpha, found {fixed}')
        with np.load(SPECTRUM_PATH) as spectrum:
            eigenvalues = np.asarray(spectrum['eigenvalues'])
            beta_proj = np.asarray(spectrum['beta_proj'])
        add_theory_order_columns(
            rows, eigenvalues, beta_proj, int(float(rows[0]['n'])))
        write_summary(rows)
        render(rows, fixed[0], int(float(rows[0]['n'])))
        logger.info('Replotted %s from %s', FIGURE_PATH, SUMMARY_PATH)
        return
    if args.train_pool_size % args.train_patches_per_image:
        raise ValueError(
            'train-pool-size must be divisible by '
            'train-patches-per-image')
    if args.population_size % args.population_patches_per_image:
        raise ValueError(
            'population-size must be divisible by '
            'population-patches-per-image')
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    total_patches = args.train_pool_size + args.population_size
    staged_path = args.scratch_dir / (
        f'vanhateren_log_gray_{args.img_size}px_{total_patches}_uint8.npy')
    staged = stage_vanhateren_uint8(
        args.source_dir, staged_path, args.train_pool_size,
        args.population_size, args.img_size,
        args.train_patches_per_image, args.population_patches_per_image,
        args.seed, args.staging_probe_images, logger,
        no_progress=args.no_progress)
    if args.stage_only:
        logger.info('Stage-only run complete: %s', staged_path)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The exact d=10,000 experiment requires CUDA.')

    device = torch.device('cuda')
    logger.info('GPU: %s', torch.cuda.get_device_name(device))

    beta_np = create_disk_teacher(args.img_size, args.radius)
    beta = torch.from_numpy(beta_np).to(device)
    eigval, eigvec, eigenvalues, beta_proj = compute_population_spectrum(
        staged, args.train_pool_size, args.population_size, beta_np,
        args.scratch_dir, device, logger, force=args.force_spectrum)
    beta_proj_t = torch.from_numpy(beta_proj.astype(np.float32)).to(device)
    signal_power = float((eigval * beta_proj_t.square()).sum())
    logger.info('Disk teacher norm^2=%.1f; signal power S=%.8g',
                float(beta @ beta), signal_power)

    ratios = np.asarray(
        DEFAULT_NOISE_RATIOS if args.noise_ratios is None
        else args.noise_ratios, dtype=float)
    if np.any(ratios < 0):
        raise ValueError('noise ratios must be nonnegative')
    sigmas = np.sqrt(ratios * signal_power)
    alphas_np = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)
    alphas = torch.from_numpy(alphas_np.astype(np.float32)).to(device)
    logger.info(
        'Noise curve: %d matched NSR points through %.3g; '
        'sigma range 0 to %.6g',
        len(ratios), ratios.max(), sigmas.max())
    logger.info(
        'RidgeCV alpha grid: %d values %.3g to %.3g '
        '(adjacent ratio %.4f)',
        len(alphas_np), alphas_np[0], alphas_np[-1],
        alphas_np[1] / alphas_np[0])

    train_pool = torch.from_numpy(np.asarray(
        staged[:args.train_pool_size], dtype=np.float32)).to(device)
    train_pool.div_(255.0)
    benchmark_generator = torch.Generator(device=device).manual_seed(args.seed)
    warmup_indices = torch.randperm(
        args.train_pool_size, generator=benchmark_generator,
        device=device)[:args.n]
    ridge_cv_and_fixed_fit(
        train_pool[warmup_indices], beta, float(sigmas[0]), alphas,
        args.fixed_alpha, benchmark_generator)
    torch.cuda.synchronize()
    benchmark_start = time.perf_counter()
    for _ in range(args.benchmark_trials):
        indices = torch.randperm(
            args.train_pool_size, generator=benchmark_generator,
            device=device)[:args.n]
        ridge_cv_and_fixed_fit(
            train_pool[indices], beta, float(sigmas[0]), alphas,
            args.fixed_alpha, benchmark_generator)
    torch.cuda.synchronize()
    seconds_per_trial = (
        time.perf_counter() - benchmark_start) / args.benchmark_trials
    total_trials = len(sigmas) * args.n_trials
    logger.info(
        'Exact paired-fit benchmark: %.4fs/trial; projected %.1fs '
        '(%.1fmin) for %d fits',
        seconds_per_trial, seconds_per_trial * total_trials,
        seconds_per_trial * total_trials / 60, total_trials)
    if args.benchmark_only:
        logger.info('Benchmark-only run complete')
        return

    metric_names = (
        'weight_error', 'gen_error', 'r2_gen', 'slope_gen',
        'slope_acc', 'acc_error', 'r2_acc')
    summary_rows: list[dict[str, object]] = []
    run_start = time.perf_counter()
    for case_index, sigma in enumerate(sigmas, start=1):
        sigma = float(sigma)
        cache = case_path(args, sigma)
        if cache.exists() and not args.force:
            logger.info('[%d/%d] Loading %s', case_index, len(sigmas), cache)
            cached_rows, _ = load_case(cache)
            summary_rows.extend((cached_rows['cv'], cached_rows['fixed']))
            continue

        alpha_de, de_risk_path = select_de_alpha(
            eigenvalues, beta_proj, sigma, args.n, alphas_np)
        cv_theory, cv_theory_error_pc = theory_metrics(
            eigenvalues, beta_proj, sigma, args.n, alpha_de)
        fixed_theory, fixed_theory_error_pc = theory_metrics(
            eigenvalues, beta_proj, sigma, args.n, args.fixed_alpha)
        trials = {
            policy: {
                name: np.empty(args.n_trials, dtype=float)
                for name in metric_names
            } for policy in ('cv', 'fixed')
        }
        trials['cv']['alpha_cv'] = np.empty(args.n_trials, dtype=float)
        weight_sum = {
            policy: np.zeros(len(beta_np), dtype=np.float64)
            for policy in ('cv', 'fixed')
        }
        representative: dict[str, np.ndarray] = {}
        error_pc_sum = {
            policy: torch.zeros_like(eigval)
            for policy in ('cv', 'fixed')
        }
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 1000 * case_index)
        iterator = range(args.n_trials)
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    iterator, desc=f'Van Hateren paired sigma={sigma:g}',
                    unit='trial')
            except ImportError:
                pass
        case_start = time.perf_counter()
        for trial_index in iterator:
            indices = torch.randperm(
                args.train_pool_size, generator=generator,
                device=device)[:args.n]
            cv_weight, fixed_weight, selected_alpha = ridge_cv_and_fixed_fit(
                train_pool[indices], beta, sigma, alphas,
                args.fixed_alpha, generator)
            for policy, weight in (
                    ('cv', cv_weight), ('fixed', fixed_weight)):
                metrics, delta_proj = fit_metrics(
                    weight, beta, eigval, eigvec, beta_proj_t,
                    signal_power)
                for name in metric_names:
                    trials[policy][name][trial_index] = metrics[name]
                weight_np = weight.cpu().numpy()
                weight_sum[policy] += weight_np
                if trial_index == 0:
                    representative[policy] = weight_np
                error_pc_sum[policy] += delta_proj.square()
            trials['cv']['alpha_cv'][trial_index] = selected_alpha
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - case_start

        policy_rows = {
            'cv': summarize_policy(
                'cv', sigma, signal_power, alpha_de, args.n, cv_theory,
                trials['cv'], cache, elapsed, args),
            'fixed': summarize_policy(
                'fixed', sigma, signal_power, args.fixed_alpha, args.n,
                fixed_theory, trials['fixed'], cache, elapsed, args),
        }
        arrays: dict[str, np.ndarray] = {
            'alpha_grid': alphas_np,
            'de_cv_risk_path': de_risk_path,
            'cv_theory_error_per_pc': cv_theory_error_pc,
            'fixed_theory_error_per_pc': fixed_theory_error_pc,
        }
        for policy in ('cv', 'fixed'):
            arrays.update({
                f'{policy}_trial_{name}': values
                for name, values in trials[policy].items()
            })
            arrays[f'{policy}_mean_weight'] = (
                weight_sum[policy] / args.n_trials).astype(np.float32)
            arrays[f'{policy}_representative_weight'] = representative[policy]
            arrays[f'{policy}_mc_error_per_pc'] = (
                error_pc_sum[policy] / args.n_trials).cpu().numpy()
        save_case(cache, policy_rows, arrays)
        summary_rows.extend((policy_rows['cv'], policy_rows['fixed']))
        logger.info(
            '[%d/%d] sigma=%.4g NSR=%.4g done %.1fs | '
            'alpha DE/MC median %.3g/%.3g | '
            'R2acc CV DE/MC median %.4g/%.4g | fixed %.4g/%.4g',
            case_index, len(sigmas), sigma, sigma ** 2 / signal_power,
            elapsed, alpha_de,
            policy_rows['cv']['mc_alpha_cv_median'],
            policy_rows['cv']['theory_r2_acc'],
            policy_rows['cv']['mc_r2_acc_median'],
            policy_rows['fixed']['theory_r2_acc'],
            policy_rows['fixed']['mc_r2_acc_median'])

    summary_rows.sort(key=lambda row: (
        float(row['sigma']), 0 if row['policy'] == 'cv' else 1))
    add_theory_order_columns(
        summary_rows, eigenvalues, beta_proj, args.n)
    write_summary(summary_rows)
    render(summary_rows, args.fixed_alpha, args.n)
    logger.info('Finished paired simulation in %.1fs',
                time.perf_counter() - run_start)
    logger.info('Summary: %s', SUMMARY_PATH)
    logger.info(
        'Raw case caches: %s',
        require_bulk_path(
            CASE_DIR, 'Van Hateren fixed-vs-CV case caches'))
    logger.info('Figure: %s', FIGURE_PATH)


if __name__ == '__main__':
    main()
