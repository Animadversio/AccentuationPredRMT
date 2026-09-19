"""Resolve bulk experiment artifacts outside the Git working tree."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_RELATIVE_ROOT = Path('Projects/AccentuationPredRMT')
BULK_ARCHIVE_NAME = 'repo_bulk_v1'


def project_store_root() -> Path:
    """Return the configured project store, failing instead of using HOME."""
    store_dir = os.environ.get('STORE_DIR')
    if not store_dir:
        raise RuntimeError(
            'STORE_DIR is required for bulk experiment artifacts; set it or '
            'pass an explicit cache path to the command.')
    return Path(store_dir).expanduser() / PROJECT_RELATIVE_ROOT


def configured_bulk_path(relative_path: str | Path) -> Path | None:
    """Return a bulk path when STORE_DIR is configured, otherwise ``None``.

    This lazy form is safe at module import time. Commands must call
    :func:`require_bulk_path` before reading or writing the returned value.
    """
    store_dir = os.environ.get('STORE_DIR')
    if not store_dir:
        return None
    return (
        Path(store_dir).expanduser() / PROJECT_RELATIVE_ROOT /
        BULK_ARCHIVE_NAME / Path(relative_path))


def require_bulk_path(path: Path | None, purpose: str) -> Path:
    """Validate a lazily resolved bulk path with an actionable error."""
    if path is None:
        raise RuntimeError(
            f'STORE_DIR is required for {purpose}; set STORE_DIR or pass an '
            'explicit path on the command line.')
    return Path(path)
