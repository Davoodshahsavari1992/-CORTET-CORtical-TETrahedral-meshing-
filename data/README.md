# Data and results in this repository

The repository ships the code, the derived measurements needed to reproduce the paper's cohort figures,
and one worked example of the surface-smoothing stage. It does **not** ship the meshes themselves or the
input surfaces, for the reasons set out below.

## What is included

| Path | What it is |
|---|---|
| `results/cohort_quality_194.csv` | Per-subject mesh quality for the full cohort: hemisphere, gestational age, node count, tetrahedron count, mean quality and worst-element quality. 194 rows. Reproduces the cohort table and the quality-versus-gestational-age figure. |
| `results/pooled_hist.csv` | Pooled per-element quality distribution across the cohort, as plotted in the paper. |
| `data/example_surface_smoothing/` | Metrics from the surface-smoothing stage on one subject: noise removed, cortical-area change and mean displacement at each iteration count. |

`cohort_quality_194.csv` carries no subject or session identifiers. Gestational age is retained because
the cohort results are reported against it; every figure in the paper can be regenerated from these
columns alone.

## What is not included, and why

**The tetrahedral meshes.** These run from roughly 16 MB to over 80 MB each, and the cohort is 194
subjects. That is far beyond what belongs in a Git repository. They are retained on institutional
storage and are available from the authors on request.

**The input cortical surfaces.** These are derived from the developing Human Connectome Project (dHCP).
dHCP data is released under a data use agreement, and derived surfaces remain subject to it, so they
cannot be redistributed here. Obtain the source data from the dHCP directly, under its own terms.


## Reproducing the pipeline on your own data

The pipeline takes a closed, orientable triangulated cortical surface in GIfTI format. Nothing in it is
specific to the dHCP beyond that: any genus-zero cortical surface at a comparable resolution should
work. Start from `cortet/surface/` for the smoothing stage, then `cortet/generation/`, then optionally
`cortet/postmesh/`. See the top-level `README.md`.

If you reproduce the cohort study, note that per-subject smoothing iteration counts are set by
gestational age under limits on area loss and surface displacement; those limits, and the measured cost
of each iteration count, are illustrated in `data/example_surface_smoothing/`.
