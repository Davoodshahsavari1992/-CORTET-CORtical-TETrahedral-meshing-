# Figures in the README

All five are renders of real CORTET meshes and of the files in `results/`; none is a schematic. They were
made for the MICCAI 2026 (CBM XXI) talk and poster and are reproduced here at web size (1760 px wide, shown
at 880 px). The scripts that made them live with the project's publication material, not in this
repository, because they depend on dHCP-derived meshes that are not distributed.

| File | What it shows | Made from |
|---|---|---|
| `cortet_overview.png` | The week-22 brain's surface mesh, with a close-up of its triangles; the CORTET mesh of the same brain cut open, with a close-up of tetrahedra from the cut face, each shrunk slightly towards its own centre so they read as separate elements | the week-22 CORTET mesh at `h = 0.6` mm (415,497 tetrahedra) |
| `cortet_weeks_21_to_38.gif` | One real fetal brain per week from 21 to 38 weeks of gestation, one camera and one world scale; different weeks are different fetuses | CORTET meshes of dHCP subjects at `h = 0.6` mm |
| `cortet_bad_elements.png` | Elements over the limit (`q > 0.6`) on the same week-22 surface: TetGen (`pq1.2/20Y`), the CGAL fill alone, and CORTET | TetGen (`pq1.2/20Y`), the CGAL fill alone and CORTET, each run on the week-22 surface |
| `cortet_worst_element.png` | The CGAL fill cut open through its worst element and coloured by `q`; that element with the elements sharing a node with it, before (`q = 0.765`) and after (`q = 0.362`) CORTET, on one camera | the CGAL-only and the final mesh of the week-22 brain, which share connectivity |
| `cortet_results.png` | Six of the meshes from 21 to 38 weeks; the pooled per-element quality of all 194 meshes after the fill alone and after CORTET; the worst element of each brain against gestational age | `results/pooled_hist.csv`, `results/cohort_quality_194.csv` |

Quality is `meshtool`'s `tet_qmetric_volume`, as everywhere in this repository: `0` for a regular tetrahedron,
`1` for a flat one, with the limit at `0.6`.
