# Results

Measured output. Every quality number here is `meshtool`'s `tet_qmetric_volume`, on which 0 is a
regular tetrahedron and 1 a degenerate one, and the solver-ready criterion is `q_max < 0.6`.


| File | What |
|---|---|
| `cohort_quality_194.csv` | per-subject quality, the 194-subject cohort at `h = 0.6` |
| `pooled_hist.csv` | per-element quality pooled over 2.0 × 10⁸ elements, three configurations |
| `table2_h_resolution_study.csv` | the paper's Table 2: three subjects × three resolutions |
| `table2_full_matrix.csv` | the wider regeneration matrix that Table 2 was checked against |
| `postmesh_cleaning_study.csv`, `postmesh_cycles_study.csv` | why the re-clean is two passes at 0.7 and 0.5, and why a third at 0.3 is refused |
| `pre_vs_post_smoothing_grid.csv`, `surface_smoothing_level_study.csv` | input smoothing level against mesh-boundary roughness and quality |
| `figures/` | boundary renders, before and after post-mesh smoothing |

---

## `table2_full_matrix.csv` — read the caveats before using it

Three subjects × three cell sizes × raw/smoothed input × with/without post-mesh smoothing, run with
the **default `--method surfacepreserving`**. 44 rows, filed verbatim as the run wrote them so they
can be checked against the logs. Reproduce a row with `run/02` at the stated `--cell-size`, then
`run/05` on its output.

**Rows where `scenario = without_postsmoothing` carry `nodes` and `tets`; the
`with_postsmoothing` rows do not** — smoothing does not change the node or element count, so the
aggregator leaves those fields empty rather than repeating them. The roughness and displacement
columns are the reverse: they only appear on the `with_postsmoothing` rows, because they are measured
against the input surface after the smooth-and-re-clean cycle.

---

## What this matrix independently confirms

**Table 2.** Every `smoothed` / `without_postsmoothing` row reproduces its Table 2 counterpart, for
the eight cells whose task completed on the correct session:

| Subject | `h` | matrix (nodes / tets / `q̄` / `q_max`) | Table 2 |
|---|---|---|---|
| `subject_GA21.86_left` | 0.4 | 194,632 / 1,101,843 / 0.1284 / 0.5497 | 0.128 / 0.550 |
| | 0.6 | 59,174 / 326,083 / 0.1289 / 0.4621 | 0.129 / 0.462 |
| | 0.8 | 25,722 / 137,933 / 0.1297 / 0.4578 | 0.130 / 0.458 |
| `subject_GA28.00_left` | 0.4 | 458,388 / 2,612,132 / 0.1284 / 0.5216 | 0.128 / 0.522 |
| | 0.6 | 139,143 / 773,954 / 0.1282 / 0.5036 | 0.128 / 0.504 |
| | 0.8 | 60,114 / 326,436 / 0.1289 / 0.5084 | 0.129 / 0.508 |
| `subject_GA33.86_left` | 0.6 | 290,248 / 1,604,834 / 0.1289 / 0.5231 | 0.129 / 0.523 |
| | 0.8 | 125,749 / 677,789 / 0.1294 / 0.5162 | 0.129 / 0.516 |

**Section 4.4.** The `subject_GA21.86_left`, `h = 0.4`, raw-input pair reproduces the published
post-mesh-smoothing result on a separate run — and, notably, with a **different smoother**: the
published figures were measured with Workbench, this matrix used surface-preserving.

| | published | this matrix |
|---|---|---|
| boundary roughness | 0.191 → 0.134 | 0.19143 → 0.13261 |
| worst element `q_max` | 0.488 → 0.500 | 0.4879 → 0.49986 |
| boundary displacement, mean | 0.175 mm | 0.17563 mm |
| boundary displacement, max | 0.480 mm | 0.46665 mm |

`q_max` and the pre-smoothing roughness agree to four significant figures. Roughness after smoothing
and mean displacement agree to about 1 per cent. Maximum displacement is 2.8 per cent lower here than
in the original measurement, the only one of the four a careful reader would call different.

That two different smoothers land this close is a stronger result than a same-method repeat would
have been: the ~30 per cent roughness reduction and the recovery to `q_max ≈ 0.50` are properties of
the smooth-and-re-clean cycle, not of one particular filter. It does mean, though, that the published
figures have still not been reproduced **with the method that produced them**: that would need
`run/05 --method wb` at `h = 0.4` on the same subject.

**A result the paper does not claim, visible here:** post-mesh smoothing drives `q_max` to almost
exactly 0.50 in nearly every one of the 22 smoothed configurations, whatever the subject, resolution
or input. That is the two-pass clean at threshold 0.5 doing precisely what it is asked to do, and it
is a good sign — the recovery is not subject-specific luck.

---

## Identifiers

Subjects are named by gestational age (`subject_GA21.86_left`), not by dHCP subject or session ID.
The GA is retained because every result here is reported against it.
