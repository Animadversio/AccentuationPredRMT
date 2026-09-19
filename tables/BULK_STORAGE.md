# Bulk experiment storage

Large per-trial arrays and case caches are stored outside this repository at:

```text
$STORE_DIR/Projects/AccentuationPredRMT/repo_bulk_v1/tables/
```

The archive contains 169 files (364,038,575 bytes at migration time) and a
`MANIFEST.sha256` file whose paths are relative to the `tables/` directory.
It includes:

- `ffhq_disk_teacher_cases/`
- `powerlaw_teacher_alignment_cases/`
- `vanhateren_fixed_vs_cv_cases/`
- `vanhateren_null_rotation_cases.npz`
- `vanhateren_null_rotation_dimension_cases.npz`
- `vanhateren_p500_null_family_cases.npz`
- `vanhateren_top_pc_tail_rotation_cases.npz`

To verify the archive:

```bash
cd "$STORE_DIR/Projects/AccentuationPredRMT/repo_bulk_v1/tables"
sha256sum --check ../MANIFEST.sha256
```

Small plot-ready summaries and final figures remain versioned in the
repository. Scripts resolve bulk paths through `scripts.storage_paths` and
fail with an actionable error when `STORE_DIR` is not configured.
