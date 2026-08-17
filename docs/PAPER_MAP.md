# Paper → code map

Every substantive claim the manuscript makes about the pipeline, and the file in this repository
that backs it. Written so a referee, or anyone re-running the work, can go from a sentence in the
paper to the code that produced it without guessing.

Every stage is runnable on its own from `run/`; the README's *Running it, stage by stage* section
says which stage reproduces which part of the paper.

## Methods §3.1 — Pipeline overview

| Paper says | Where |
|---|---|
| Six sequential stages, S1–S6 | `cortet/generation/generate_tetrahedral_mesh.py`, `run_pipeline()` |
| Stages 3, 4 and 6 are optional and bypassable | `--skip-gmsh`, `--skip-meshtool`; S6 runs only under `--reproject` |
| Implemented in Python, chaining CGAL, Gmsh, `meshtool`, with NiBabel / meshio / Trimesh for I/O | imports in the same file |
| Input surfaces are GIfTI | `stage1_generate_volume_mesh()` loads via `nibabel` |

## Methods §3.2 — Volume generation and quality optimisation (S1–S4)

| Paper says | Where |
|---|---|
| Inconsistently oriented triangles repaired, temporary STL exported | `surface.fix_normals()` then `surface.export(tmp_stl)` |
| Cell size `h` bounds each tet's circumradius **and** the surface Delaunay ball radius; facet distance is `h/2` | `max_cell_circumradius=cell_size`, `max_radius_surface_delaunay_ball=cell_size`, `max_facet_distance=cell_size/2.0` |
| Four CGAL optimisation passes: ODT, Lloyd, perturb, then sliver removal last | `odt=True, lloyd=True, perturb=True, exude=True` |
| Refinement replaces every input vertex, so per-vertex fields must be re-interpolated | `cortet/fields/map_surface_fields_to_mesh.py` |
| Gmsh Relocate3D global vertex smoothing, ×5 | `stage3_gmsh_relocate(..., n_passes=5)`, `--gmsh-passes` default 5 |
| `meshtool` cleaning, three passes at `q > 0.7, 0.5, 0.3` | `stage4_meshtool_clean(..., thresholds=[0.7, 0.5, 0.3])` |
| Cleaning repositions vertices; element count and connectivity unchanged; nothing deleted or remeshed | `meshtool clean quality -thr=...` |

## Methods §3.3 — Solver-ready output (S5–S6)

| Paper says | Where |
|---|---|
| Boundary-face list reconstructed from tetrahedral connectivity | `write_braingrowth_mesh()`, faces belonging to exactly one tet |
| Surface nodes ordered contiguously ahead of interior nodes | same function, step 3; independently checkable with `cortet/verify/check_node_reordering.py` |
| Node ordering and signed-volume conventions matched to the solver | the n3↔n4 swap in `write_braingrowth_mesh()` |
| S6 snaps boundary nodes to the nearest point of the input surface, re-checking inversion | `stage6_reproject_boundary()`, which counts non-positive signed volumes afterwards |
| Mesh can also be exported to Abaqus, FEBio, FEniCS, VTK or Gmsh | `cortet/export/export_mesh_formats.py` |

## Methods §3.5 — Pre-processing

| Paper says | Where |
|---|---|
| Surface-preserving Laplacian filter, constrained across edges above a dihedral-angle threshold | `cortet/surface/smooth_one_multi_N.py`, `--angle` |
| Four smoothers compared across 460 fetal surfaces, GA 20.9–38.3 | that script's docstring |
| Iterations set per subject by gestational age, under limits on area loss and displacement | the GA bands in the same docstring; `run/01_smooth_surface.py --ga` |
| Preservation confirmed by tracking curvature and area | the metrics CSV the script writes; example in `data/example_surface_smoothing/` |
| Resampling with `wb_command -surface resample` from ico-6 (40,962 vertices) | **not in this repository** — upstream of the pipeline, done with Connectome Workbench and MSM |

## Methods §3.6 — Evaluation

| Paper says | Where |
|---|---|
| Quality measured with `meshtool`'s `tet_qmetric_volume`, the same criterion S4 repairs against | `query_meshtool_quality()`; S4 uses the same measure |
| Solver-ready criterion `q_max < 0.6` | stated throughout; `results/cohort_quality_194.csv` has no subject above it |

## Results §4.1 — Cohort ablation (Table 1, Fig. 2)

| Paper says | Where |
|---|---|
| N = 194 left-hemisphere subjects, GA 21–38, at `h = 0.6` mm | `results/cohort_quality_194.csv` — 194 rows, all left, GA 21.43–38.29 |
| 2.0 × 10⁸ tetrahedra per configuration | `results/pooled_hist.csv` — 200,379,097 per stage |
| CGAL only: `q̄ = 0.135`, median `q_max = 0.780` | Table 1; per-element distribution in `pooled_hist.csv` |
| CGAL only leaves ≈1.7 × 10⁵ elements above `q > 0.6` | `pooled_hist.csv` — 173,073 |
| + Gmsh: median `q_max = 0.732`, tail reduced but not resolved | `pooled_hist.csv` — 36,350 elements above threshold |
| Full pipeline: not one element above `q = 0.6`, in any subject | `pooled_hist.csv` — 0 above threshold; highest occupied bin ends at 0.565 |
| Full pipeline `q_max` median 0.508, range 0.462–0.561 | `cohort_quality_194.csv` — median 0.5083, range 0.4621–0.5613 |
| Mean quality varies by less than 0.01 across configurations | `cohort_quality_194.csv` — `q̄` spans 0.1280–0.1294 |
| TetGen out-of-the-box: `q̄ = 0.242`, `q_max = 0.996`, median 15,955 elements over threshold | produced by the ablation driver; **per-subject TetGen rows are not shipped** |
| TetGen given the SAME S3 + S4 stages still does not reach the target (answers R1.5) | `run/04_tetgen_baseline.py --with-s3s4`; `q_max` reported to stay ≈0.98–1.00 |
| Tallinen 22-week mesh: `q̄ = 0.162`, `q_max = 0.811`, 15 elements over threshold | third-party mesh, measured with the same tool; not redistributable |

## Results §4.2 — Cohort and development (Fig. 3)

| Paper says | Where |
|---|---|
| `q_max` held below 0.6 across GA 21–38, range 0.46–0.56 | `cohort_quality_194.csv` |
| `q̄` flat at ≈0.128–0.129 | same file |
| Mesh grows from ~0.3 to ~2.1 million tetrahedra | same file — 306,571 to 2,114,138 |

## Results §4.3 — Effect of resolution (Table 2)

| Paper says | Where |
|---|---|
| Three subjects at `h` = 0.4 / 0.6 / 0.8; element count scales ≈ `h⁻³`; quality flat | `results/table2_h_resolution_study.csv` |
| — eight of the nine cells | **independently reproduced** by the wider regeneration matrix, node and tet counts identical and `q_max` to 3 decimals (`results/table2_full_matrix.csv`) |
| — the `subject_GA21.86_left`, `h = 0.6` row specifically | the real mesh file has **59,174 nodes / 326,083 tets**, matching the published row exactly |
| — of which eight rows | marked `confirmed` in that CSV |

Regeneration: `run/02_generate_mesh.py` at each `--cell-size`, then `run/05` on each output.

## Results §4.4 — Post-mesh smoothing, and the simulation (Fig. 5)

| Paper says | Where |
|---|---|
| The pipeline provides optional post-mesh boundary smoothing with the cleaning it requires | `cortet/postmesh/run_full_postmesh_pipeline.py` |
| Roughness falls 0.191 → 0.130 at `h = 0.4` (≈32%) | `results/table2_full_matrix.csv`, smoothed-input `h = 0.4` row |
| `q_max` improves 0.550 → 0.534 | same row |
| Boundary stays within 0.173 mm mean / 0.434 mm max | same row |
| Smoothing alone leaves `q_max = 0.921`, so re-cleaning is not optional | `cortet/postmesh/run_full_postmesh_pipeline.py` |
| Two cleaning passes, 0.7 then 0.5, and not a third at 0.3 | `--thresholds` default `[0.7, 0.5]`; §6 records the failed third pass |
| Solvers that do not resolve a free boundary can bypass the step | the step is a separate entry point, not part of generation |

Before and after renders of the boundary this step acts on: `results/figures/`.

**As of the 31 July manuscript, all six Section 4.4 figures are the smoothed-input `h = 0.4` row of
`results/table2_full_matrix.csv`, quoted to three decimal places.** They are shipped here, so this
section is backed by data in the repository rather than by a decision log.

The 36-combination regeneration matrix (three subjects × three `h` × raw/smoothed input) corroborates
these with `--method surfacepreserving`: post-smoothing `q_max` in 0.494–0.503 across 35 of 36 cells,
roughness 0.130–0.150. See the README's *Verification status*. The published figures themselves were
measured with `--method wb`, so use that flag to reproduce them exactly.

## Discussion — the five silent requirements

The paper names five requirements that mesh generators do not enforce and that fail silently. Each is
enforced here, and each is independently checkable:

All five are applied and then re-verified in one command by `run/06_export_braingrowth.py`, which
exits non-zero if any check fails. Run on a real 326,083-tet pipeline mesh
(`subject_GA21.86_left`, h = 0.6) all five pass. **verified on real data**

| Requirement | Enforced by |
|---|---|
| Input vertices are discarded, so fields must be re-interpolated, not indexed | `cortet/fields/map_surface_fields_to_mesh.py` |
| A boundary-face list must be reconstructed | `write_braingrowth_mesh()`; `cortet/postmesh/finalize_for_braingrowth.py` |
| Surface nodes must precede interior nodes | `write_braingrowth_mesh()` step 3 |
| Per-element node ordering must match the solver | the n3↔n4 swap |
| The signed-volume convention must match the solver | same |

## Code availability statement

The statement scopes the release to the meshing pipeline and excludes the FEM solver, pointing at
Wang et al. (2021) and `github.com/rousseau/BrainGrowth`. That matches this repository: no solver is
included. Per-subject quality measurements are in `results/`; the meshes themselves are not shipped,
under the dHCP data use terms — see `data/README.md`.

---

## What this repository does **not** contain

Stated here so the omissions are deliberate and visible rather than discovered:

- **The FEM solver.** By design; see above.
- **The MSM / `wb_command` surface-resampling front end.** Upstream third-party tooling.
- **Any dHCP-derived surface or mesh.** Data use terms.
- **Per-subject TetGen and Tallinen measurements** behind two rows of Table 1. The driver that
  produced them runs on CREATE; only the aggregate reached the paper.
  "all rights reserved" by default.
