"""
Data spectrum utilities for ridge regression theory validation.

Provides three spectrum types:
  - isotropic: Σ = I  (all eigenvalues = 1)
  - powerlaw:  λ_k = k^{-alpha}  (normalized)
  - vanhateren: empirical covariance of natural image patches from the Van Hateren dataset

Each function returns (eigenvalues, eigenvectors) where:
  - eigenvalues : np.ndarray, shape (d,), sorted descending
  - eigenvectors: np.ndarray, shape (d, d), columns are eigenvectors u_k
"""
import os
import numpy as np


VANHATEREN_DIR = (
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
    "Datasets/vanhateren_natural_stimuli"
)


# ──────────────────────────────────────────────
# Isotropic
# ──────────────────────────────────────────────

def isotropic_spectrum(d):
    """Σ = I: all eigenvalues equal to 1, eigenvectors = canonical basis.

    Returns
    -------
    eigenvalues : np.ndarray, shape (d,)   all ones
    eigenvectors: np.ndarray, shape (d, d) identity matrix
    """
    return np.ones(d), np.eye(d)


# ──────────────────────────────────────────────
# Power-law
# ──────────────────────────────────────────────

def powerlaw_spectrum(d, alpha=1.0, normalize=True):
    """λ_k = k^{-alpha}, k = 1..d.

    Parameters
    ----------
    d : int
    alpha : float  spectral decay exponent
    normalize : bool  if True, normalize so mean eigenvalue = 1

    Returns
    -------
    eigenvalues : np.ndarray, shape (d,)   descending order
    eigenvectors: np.ndarray, shape (d, d) random orthogonal (seeded to None → caller provides)

    Note: eigenvectors are returned as identity. For a random orientation,
    pass the eigenvalues to your simulation and draw a random Q via QR.
    """
    k = np.arange(1, d + 1, dtype=float)
    lam = k ** (-alpha)
    if normalize:
        lam = lam / lam.mean()
    return lam, np.eye(d)


# ──────────────────────────────────────────────
# Van Hateren natural image patches
# ──────────────────────────────────────────────

def _load_vh_image(path):
    """Load a single Van Hateren .iml file → float array shape (1024, 1536)."""
    with open(path, 'rb') as f:
        raw = f.read()
    img = np.frombuffer(raw, dtype='>u2').astype(float)  # big-endian uint16
    img = img.reshape(1024, 1536)
    return img


def _extract_patches(img, patch_hw, n_patches, rng):
    """Extract n_patches random patches of shape (patch_hw, patch_hw) from img."""
    H, W = img.shape
    ph = patch_hw
    rows = rng.integers(0, H - ph, size=n_patches)
    cols = rng.integers(0, W - ph, size=n_patches)
    patches = np.stack([img[r:r+ph, c:c+ph] for r, c in zip(rows, cols)])
    return patches  # (n_patches, ph, ph)


def vanhateren_spectrum(patch_hw=8, n_patches_per_image=200, n_images=50,
                        log_luminance=True, center=True, rng=None,
                        top_k=None, use_randomized_svd=False, svd_oversampling=10):
    """Compute eigenspectrum of Van Hateren natural image patches.

    Loads .iml files from VANHATEREN_DIR, extracts patches, computes the
    empirical covariance, and returns its eigendecomposition.

    Parameters
    ----------
    patch_hw : int
        Patch height/width in pixels. Dimension d = patch_hw².
    n_patches_per_image : int
        Number of patches sampled per image.
    n_images : int
        Number of images to use.
    log_luminance : bool
        Apply log(1 + x) to pixel values before extracting patches.
    center : bool
        Subtract mean patch before computing covariance.
    rng : np.random.Generator or None
    top_k : int or None
        If set, return only the top-k eigenvectors (via randomized SVD).
        Automatically set to min(d, N//2) if use_randomized_svd=True.
    use_randomized_svd : bool
        Use torch randomized SVD instead of full eigh. Required for d > ~3000.
    svd_oversampling : int
        Extra columns for randomized SVD (larger = more accurate, slower).

    Returns
    -------
    eigenvalues : np.ndarray, shape (d,) or (top_k,)   descending
    eigenvectors: np.ndarray, shape (d, d) or (d, top_k) columns = eigenvectors
    patches_flat : np.ndarray, shape (N, d)  the raw (centered) patches
    """
    if rng is None:
        rng = np.random.default_rng(0)

    iml_files = sorted([
        os.path.join(VANHATEREN_DIR, f)
        for f in os.listdir(VANHATEREN_DIR)
        if f.endswith('.iml')
    ])
    if len(iml_files) == 0:
        raise FileNotFoundError(
            f"No .iml files found in {VANHATEREN_DIR}")
    n_images = min(n_images, len(iml_files))
    chosen = rng.choice(len(iml_files), size=n_images, replace=False)
    chosen_files = [iml_files[i] for i in sorted(chosen)]

    all_patches = []
    for fpath in chosen_files:
        img = _load_vh_image(fpath)
        if log_luminance:
            img = np.log1p(img)
        patches = _extract_patches(img, patch_hw, n_patches_per_image, rng)
        all_patches.append(patches.reshape(n_patches_per_image, -1))

    patches_flat = np.concatenate(all_patches, axis=0).astype(np.float32)  # (N, d)
    N, d = patches_flat.shape

    if center:
        patches_flat -= patches_flat.mean(axis=0, keepdims=True)

    # For large d, use randomized SVD on the data matrix directly (avoids d×d cov)
    # The covariance is patches_flat.T @ patches_flat / (N-1)
    # Its eigendecomposition = V @ diag(σ²/(N-1)) @ Vᵀ from the SVD of patches_flat
    if use_randomized_svd or (d > 2000):
        import torch
        k = top_k if top_k is not None else min(d, N) - 1
        k = min(k, min(d, N) - 1)
        print(f"  Randomized SVD: d={d}, N={N}, k={k} ...")
        X_t = torch.tensor(patches_flat, dtype=torch.float32)
        if torch.cuda.is_available():
            X_t = X_t.cuda()
        # torch.linalg.svd is too slow for huge matrices; use torch.svd_lowrank
        # torch.svd_lowrank returns (U, S, V) where V is (d, q) — columns are right sing. vecs
        U, S, V = torch.svd_lowrank(X_t, q=k + svd_oversampling)
        eigvals = (S[:k] ** 2 / (N - 1)).cpu().numpy().astype(float)
        eigvecs = V[:, :k].cpu().numpy().astype(float)  # (d, k)
    else:
        # Full eigendecomposition of d×d covariance
        cov = patches_flat.T @ patches_flat / (N - 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(float))
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        if top_k is not None:
            eigvals = eigvals[:top_k]
            eigvecs = eigvecs[:, :top_k]

    return eigvals, eigvecs, patches_flat


def get_spectrum(name, d=64, alpha=1.0, rng=None, **kwargs):
    """Unified interface. Returns (eigenvalues, eigenvectors).

    Parameters
    ----------
    name : str   one of 'isotropic', 'powerlaw', 'vanhateren'
    d : int      dimension (used for isotropic/powerlaw; ignored for vanhateren)
    alpha : float  power-law exponent
    rng : np.random.Generator

    Returns
    -------
    eigenvalues : (d,)
    eigenvectors: (d, d)   random orthogonal for isotropic/powerlaw
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if name == 'isotropic':
        lam, _ = isotropic_spectrum(d)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return lam, Q

    elif name == 'powerlaw':
        lam, _ = powerlaw_spectrum(d, alpha=alpha)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return lam, Q

    elif name == 'vanhateren':
        patch_hw = kwargs.pop('patch_hw', 8)  # 8x8 → d=64
        lam, eigvecs, _ = vanhateren_spectrum(patch_hw=patch_hw, rng=rng, **kwargs)
        return lam, eigvecs

    else:
        raise ValueError(f"Unknown spectrum: {name!r}. "
                         "Choose from 'isotropic', 'powerlaw', 'vanhateren'.")
