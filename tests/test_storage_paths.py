from pathlib import Path

import pytest

from scripts.storage_paths import (
    BULK_ARCHIVE_NAME,
    PROJECT_RELATIVE_ROOT,
    configured_bulk_path,
    project_store_root,
    require_bulk_path,
)


def test_project_store_root_uses_store_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE_DIR", str(tmp_path))

    assert project_store_root() == tmp_path / PROJECT_RELATIVE_ROOT


def test_configured_bulk_path_is_lazy_without_store_dir(monkeypatch):
    monkeypatch.delenv("STORE_DIR", raising=False)

    assert configured_bulk_path("tables/cases.npz") is None
    with pytest.raises(RuntimeError, match="STORE_DIR is required"):
        require_bulk_path(None, "test cases")


def test_configured_bulk_path_uses_versioned_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE_DIR", str(tmp_path))

    expected = (
        tmp_path
        / PROJECT_RELATIVE_ROOT
        / BULK_ARCHIVE_NAME
        / Path("tables/cases.npz")
    )
    assert configured_bulk_path("tables/cases.npz") == expected
