# Figure index

Figures are grouped by the scientific question they address. Dataset names
(`ffhq`, `vanhateren`, `powerlaw`) remain in filenames so related analyses can
be compared within one folder.

| Folder | Analysis / contents |
| --- | --- |
| `ridge_error/` | Per-PC RMT-vs-Monte-Carlo validation, large-dimensional checks, and noise sweeps |
| `accentuation/` | Generalization-versus-accentuation error, optimal regularization, and alignment diagnostics |
| `peer_validation/` | Own-path and independent-peer error/R² validation |
| `model_selection/` | CV-selected regularization, fixed-vs-CV comparisons, and spectral mismatch diagnostics |
| `teacher_alignment/` | Power-law teacher placement and eigenbasis-weight analyses |
| `natural_image_disk_teacher/` | Core FFHQ disk-teacher validation, weights, and eigenspectrum analyses |
| `feature_space/` | Whitening/top-PC/interpolated feature-space analyses for FFHQ and Van Hateren |
| `null_rotations/` | Prediction-null rotations, dimensional scaling, and top-PC-to-tail perturbations |
| `geometry/` | Iso-error geometry and ridge-path illustrations (PNG/PDF pairs) |
| `theory/` | Theory/summary diagrams and theory-vs-simulation checks |
| `vanhateren_landscape/` | Van Hateren error/R²/slope landscapes and one-dimensional slices |

Plot-ready numerical results belong in `tables/`; notebook-local previews and
run-specific diagnostics remain under `notebooks/outputs/`. New publication or
report figures should be written to one of the topic folders above rather than
directly into `figures/`.

For backward compatibility, figures that previously lived directly under
`figures/` retain relative symbolic-link aliases at their original paths.
This keeps old notebooks, reports, and agent conversations resolvable while
the canonical image remains in exactly one topic folder. New figures should
not add top-level aliases unless an old path already exists in published or
shared material.
