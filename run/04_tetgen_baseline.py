#!/usr/bin/env python3
r"""STAGE 04 -- TetGen baseline.

TetGen out-of-the-box on the same input surface, measured the same way: the
controlled comparison that isolates the generator from the geometry. Switches
are pq1.2/20Y on a PyVista surface prepared with consistent, auto-oriented
normals -- the configuration behind the published baseline. Changing them
breaks comparability.

--with-s3s4 pushes TetGen's output through CORTET's own S3 and S4, asking
whether the optimisation stages rather than the choice of tetrahedraliser carry
the result.

If a quality value comes back outside [0, 1] the input is malformed rather than
merely poor. TetGen run without the normal-consistency prep returned q_max = 2.0
here; cortet/common.py flags it explicitly.

    python3 run/04_tetgen_baseline.py --surface subject_sp30.surf.gii --out-dir ablation/
    python3 run/04_tetgen_baseline.py --surface subject_sp30.surf.gii --out-dir ablation/ --with-s3s4

WHAT TO SEND BACK from a cluster run: stage04_tetgen.csv and the console log.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


# The switches used for the paper's baseline. Keep them: changing them makes the
# comparison no longer like-for-like with the published TetGen row.
#   p  tetrahedralise a piecewise linear complex
#   q1.2/20  radius-edge ratio 1.2, minimum dihedral angle 20 degrees
#   Y  preserve the input surface mesh (do not add Steiner points on it)
TETGEN_SWITCHES = "pq1.2/20Y"

PUBLISHED = dict(qbar=0.242, qmax=0.996, qmax_lo=0.974, qmax_hi=1.000,
                 over=15955, over_lo=12095, over_hi=35414)


def tetrahedralise(surface, switches):
    """TetGen out-of-the-box on a GIfTI surface. Returns (points, tets)."""
    import numpy as np
    import pyvista as pv
    import tetgen

    vertices, faces = C.read_gii_surface(surface)
    # pyvista wants faces as [3, i, j, k, 3, i, j, k, ...]
    padded = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
    poly = (pv.PolyData(vertices, padded)
            .compute_normals(auto_orient_normals=True, consistent_normals=True)
            .clean())

    t = tetgen.TetGen(poly)
    t.tetrahedralize(switches=switches)
    return (np.asarray(t.grid.points, dtype=np.float64),
            np.asarray(t.grid.cells_dict[10], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meshtool", default=None)
    ap.add_argument("--switches", default=TETGEN_SWITCHES,
                    help=f"TetGen switches (default {TETGEN_SWITCHES}, as used for "
                         f"the published baseline; changing them breaks "
                         f"comparability)")
    ap.add_argument("--with-s3s4", action="store_true",
                    help="after TetGen, also apply this pipeline's OWN S3 (Gmsh "
                         "Relocate3D) and S4 (meshtool 3-pass clean) to the TetGen "
                         "output, and report quality at all three checkpoints. "
                         "This asks whether the optimisation stages, rather than "
                         "the choice of tetrahedraliser, carry the result")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    C.banner("STAGE 04 — TetGen baseline")

    try:
        import tetgen           # noqa: F401
        import pyvista          # noqa: F401
    except ImportError as exc:
        C.banner("UNAVAILABLE")
        print(f"  This stage needs the tetgen and pyvista packages: {exc}\n")
        print("    pip install tetgen pyvista\n")
        print("  Nothing else in the package depends on them — stages 01, 02, 03,")
        print("  05 and 06 are unaffected.\n")
        return 2

    meshtool = C.find_meshtool(args.meshtool)
    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = args.work_dir or os.path.join(args.out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    ok, detail = C.check_meshtool_works(meshtool, work_dir)
    if not ok:
        C.fail(C.meshtool_problem(meshtool, detail) +
               " Run run/00_check_environment.py first.")
    C.step(f"meshtool  : {meshtool}  ({detail})")
    C.step(f"Switches  : {args.switches}")

    if args.switches != TETGEN_SWITCHES:
        C.warn(f"switches differ from the published baseline "
               f"({TETGEN_SWITCHES}); this run is no longer directly comparable "
               f"with Table 1's TetGen row")

    out_csv = os.path.join(args.out_dir, "stage04_tetgen.csv")
    write_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["surface", "switches", "nodes", "tets",
                             "qbar", "qmax", "n_above_0.6", "solver_ready"])

        for surface in args.surface:
            if not os.path.exists(surface):
                C.warn(f"skipping, not found: {surface}")
                continue

            C.banner(f"TetGen on {os.path.basename(surface)}", char="#")
            try:
                points, tets = tetrahedralise(surface, args.switches)
            except Exception as exc:                          # noqa: BLE001
                C.warn(f"TetGen failed: {type(exc).__name__}: {exc}")
                writer.writerow([os.path.basename(surface), args.switches,
                                 "FAIL", "FAIL", "", "", "", ""])
                fh.flush()
                continue

            C.step(f"TetGen produced {len(points):,} nodes, {len(tets):,} tets")
            q = C.measure_mesh(points, tets, meshtool, work_dir,
                               per_element=True, tag="s04")
            if not q:
                C.warn("measurement failed")
                continue

            C.report_quality("TetGen out-of-the-box", q, len(points), len(tets))
            passed = q["max"] < C.SOLVER_READY_THRESHOLD

            pub_qmax = "{:.4f} [{:.3f}-{:.3f}]".format(
                PUBLISHED["qmax"], PUBLISHED["qmax_lo"], PUBLISHED["qmax_hi"])
            pub_over = "{:,} [{:,}-{:,}]".format(
                PUBLISHED["over"], PUBLISHED["over_lo"], PUBLISHED["over_hi"])

            print("\n  Against the published TetGen row:")
            print(f"    {'':<18}{'measured':>12}{'published':>26}")
            print(f"    {'q_bar':<18}{q['mean']:>12.4f}{PUBLISHED['qbar']:>26.4f}")
            print(f"    {'q_max':<18}{q['max']:>12.4f}{pub_qmax:>26}")
            if q.get("n_above") is not None:
                print(f"    {'elements q>0.6':<18}{q['n_above']:>12,}{pub_over:>26}")

            print()
            if passed:
                C.warn("this TetGen run PASSED the solver-ready criterion, which "
                       "the published baseline did not. Worth reporting — it "
                       "would change what Table 1 can claim.")
            else:
                print(f"  FAIL, as expected: worst element {q['max']:.4f} >= "
                      f"{C.SOLVER_READY_THRESHOLD}.")
                print(f"  This is the comparison the paper reports — an untuned "
                      f"general-purpose\n  run leaves the sliver population a "
                      f"worst-element criterion is built to catch.")

            writer.writerow([os.path.basename(surface), args.switches,
                             len(points), len(tets), f"{q['mean']:.4f}",
                             f"{q['max']:.4f}", q.get("n_above", ""),
                             "yes" if passed else "no"])
            fh.flush()

            # ---- optionally push TetGen output through OUR S3 and S4 ----
            if args.with_s3s4:
                C.banner("TetGen output through this pipeline's own S3 and S4")
                print("  Referee question: is the result down to CGAL, or to the\n"
                      "  optimisation stages? Same downstream stages, different\n"
                      "  tetrahedraliser. Either answer is publishable.\n")
                base_carp = os.path.join(work_dir, "_tetgen_s3s4")
                C.write_carp(points, C.orient_positive(points, tets)[0], base_carp)
                rc = C.run_subprocess(
                    [sys.executable,
                     C.script_path("cortet", "tetgen", "tetgen_s3s4.py"),
                     "--carp-in", base_carp,
                     "--out-dir", os.path.join(args.out_dir, "tetgen_S3S4"),
                     "--meshtool", meshtool],
                    "S3 (Gmsh Relocate3D) then S4 (meshtool 3-pass clean)",
                    allow_fail=True)
                if not rc:
                    C.warn("the S3/S4 chain failed; the out-of-the-box row above "
                           "still stands. Check its own --help for the flags it "
                           "expects. "
                           "the SLURM version that has been exercised.")
                else:
                    print("\n  Quality at each checkpoint is printed above by "
                          "tetgen_s3s4.py.\n  For the full three-subject matrix "
                          "with aggregation against the CGAL\n  numbers, use "
                          "--with-s3s4 instead.\n")
                C.remove_carp(base_carp)

    C.banner("DONE")
    print(f"  Results: {out_csv}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
