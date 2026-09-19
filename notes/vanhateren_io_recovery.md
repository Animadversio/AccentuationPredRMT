# Van Hateren I/O recovery — 2026-09-17

## Verified scratch locations

- GPU node: `holygpu8a13203`, existing Slurm allocation `46777215`.
- Local XFS workspace: `/scratch/tmp/AccentuationPredRMT.bOcMkB` (833 GB free at inspection).
- Cross-node NFS backup: `/n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT/vanhateren_selection_estimation_local_20260917/`.
- Final archive remains `$STORE_DIR/Projects/AccentuationPredRMT/`.

## Verified progress

All 26 noise levels of Gaussian-oracle selection and all 78 independent distributional evaluations completed in approximately 5 seconds with local output, without touching Lustre inputs. NFS backup took 0.213 seconds; evaluated.csv SHA256 matched:

`7e3d343c3469ef2395af625cb126b4dc49b95144d0d0cecbea83e358171dae2d`.

Initial local recovery completed theory only. **Update: the direct-download workflow below completed all 100 real-image MC trials across 26 noise levels and 3 policies.** No Gaussian-design MC substitution was made. Small final summaries and figures are in `notebooks/outputs/pixel_ridge/vanhateren_selection_estimation/`.

### Direct-download recovery

The local repository `/n/home12/binxuwang/Github/vanhateren` documents the download format and original server. The old server timed out. A working mirror is `https://pirsquared.org/research/vhatdb/full/vanhateren_iml.zip`.

Download directly into the local workspace, then run `scripts.prepare_vanhateren_download` to verify the 4167 filenames (including the original 45 missing IDs), file sizes and ZIP CRCs. The validated image directory is `/scratch/tmp/AccentuationPredRMT.bOcMkB/iml`. Pass this directory with `--source-dir` when resuming; no Lustre input access is needed. The original notebook spectrum remains unchanged, and the runner checks its signal variance against the newly reconstructed population before MC.

Download took 55 seconds; extraction and integrity checks took 54 seconds; patch staging took 9 seconds. The archive SHA256 is `7e33fc2a2dd795a85b6d8fb408424c4438a9cbcaee9b7b4d7fcf311d47226781`. Reconstructed signal variance 1736.34553 matches cached-spectrum variance 1736.34497 with relative difference 3.24e-7. The first GPU MC pilot included startup overhead (4.13 seconds); subsequent loop completed much faster. All 7800 policy/noise/trial rows were produced.

The compressed archive is backed up as `vanhateren_iml.zip` in the NFS backup directory, alongside the image manifest, 220 MB patch cache and numerical results. Expanded raw images need not be duplicated because they can be regenerated from the validated archive.

The old process received SIGTERM, but was blocked inside Lustre fstat on oracle_path_22.csv; delivery/exit can be delayed until the kernel operation returns. Do not cancel the whole Nanoclaw allocation.

## Resume when input images are readable

On the allocated GPU node, from the repository:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -u -m scripts.run_vanhateren_selection_estimation \
  --trials 100 --work-dir /scratch/tmp/AccentuationPredRMT.bOcMkB \
  --source-dir /scratch/tmp/AccentuationPredRMT.bOcMkB/iml
```

This loads selected.csv and evaluated.csv, stages the original deterministic image split to local XFS, checks the signal variance against the original spectrum, then benchmarks and runs the real-image MC. `--source-dir` can point to a verified mirror of the original .iml dataset; do not change image names/split/normalization.

If the node-local directory is lost, restore the NFS backup into a fresh local workspace first. The NFS backup is scratch, not a permanent archive.

## Archive after Lustre recovers

Use a fresh destination under the authorized project folder to avoid the previously stalled partial file. From a node with both mounts:

```bash
rsync -a --checksum \
  /n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT/vanhateren_selection_estimation_local_20260917/ \
  "$STORE_DIR/Projects/AccentuationPredRMT/vanhateren_selection_estimation_recovered_20260917/"
```

After completing any further MC, first refresh the NFS backup from the local workspace, then archive. Verify checksums before removing any temporary copies. No archive transfer has yet been completed and no automatic monitor has been installed.
