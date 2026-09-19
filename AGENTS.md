# Project working agreements

## Large experiment storage

- User-designated bulk storage: `$STORE_DIR/Projects/AccentuationPredRMT`.
- Put image caches, large arrays, per-trial simulation outputs and long-run logs under this directory, not HOME or the Git repository.
- Verify `STORE_DIR` is configured; do not silently fall back to HOME.
- During Lustre outages, the user authorizes node-local `/scratch/tmp` workspaces and non-Lustre NFS scratch at `/n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT`. Back up local results there and archive back under the designated STORE_DIR project folder after recovery. Node-local files are temporary and not cross-node accessible.
- Keep source code, notebooks, small plot-ready summaries and final figures in the repository when useful.
- Start long computations with a timing pilot, report ETA and log path, cache reusable results, and profile unreasonable runtimes before scaling.
