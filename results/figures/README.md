# Figures

## Post-mesh boundary smoothing, before and after

| | |
|---|---|
| `boundary_before_smoothing_lateral.png` | the mesh boundary straight out of generation |
| `boundary_after_smoothing_lateral.png` | the same mesh after stage 05: smoothing plus the two-pass re-clean |

Subject `subject_GA21.86_left`, GA 21.86 weeks, left hemisphere, lateral view, wireframe over the shaded
boundary surface. Both panels use one shared crop and identical dimensions (1500 × 809), so the two
are registered and can be flipped between or placed side by side without rescaling.

**What to look at.** The overall shape and the sulcal groove are the same in both — smoothing is not
reshaping the anatomy. The difference is in the boundary tessellation: the *before* panel carries the
faceting Delaunay refinement leaves behind, visible as a stippled, bumpy quality across the surface
and along the silhouette. In the *after* panel the silhouette is clean and the facets are gone, while
the fold is still there.

That distinction is the point of stage 05. For most solvers the faceting is not a defect. For a
solver that advances a growing free surface, it is a perturbation of the very surface whose
instability produces the folds.

**The numbers behind the picture** (§4.4 of the paper): boundary
roughness 0.191 → 0.134, about 30 per cent removed; worst element `q_max` 0.488 → 0.500, still well
inside the `q_max < 0.6` criterion; boundary within 0.175 mm of the input surface on average, 0.480 mm
at worst, inside the 0.5 mm imaging voxel.

Smoothing *alone*, without the re-clean, leaves `q_max = 0.921` and is unusable. These renders are of
the smoothed **and re-cleaned** mesh — the only version that should ever be shown or simulated.

## Provenance and reuse

Rendered from the mesh boundary with PyVista, and downsampled from 6400 × 5120 for distribution: cropped to a shared bounding box, resampled with Lanczos, and reduced
to a 128-colour palette. These renders are essentially a white background, one surface tint and dark
edges, so the palette reduction is visually lossless while cutting each file from about 14 MB to
under 0.5 MB. The full-resolution originals are held with the project and are available on request.

These are **renders**, not data. They show mesh geometry with no imaging, no annotation and no
identifiers, on a plain background — the same class of figure the paper itself publishes. No
dHCP-derived surface or mesh file is distributed with this repository; see `data/README.md`.
