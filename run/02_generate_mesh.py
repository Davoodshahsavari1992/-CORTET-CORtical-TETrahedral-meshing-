#!/usr/bin/env python3
r"""STAGE 02 -- full CORTET, without post-mesh smoothing.

Stages S1 to S5 of the paper's Figure 1, plus S6 under --reproject. This is the
pipeline as the cohort results were produced, and the right stage for most
target solvers. Defaults to h = 0.6 mm.

It measures the result and passes or fails it against q_max < 0.6.

    python3 run/02_generate_mesh.py --surface smoothing/subject_sp30.surf.gii \
        --out-dir meshes/ --meshtool ~/meshtool/meshtool
    python3 run/02_generate_mesh.py --surface ... --out-dir ... --cell-size 0.4
    python3 run/02_generate_mesh.py --surface ... --out-dir ... --reproject

WHAT TO SEND BACK from a cluster run: stage02_generate.csv and the console log.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


def expected_for(subject, cell_size):
    """Look up the published numbers for this subject and h, if we have them."""
    path = C.script_path("results", "table2_h_resolution_study.csv")
    if not (subject and os.path.exists(path)):
        return None
    for row in csv.DictReader(open(path)):
        if row["subject"] in subject and abs(float(row["h_mm"]) - cell_size) < 1e-9:
            return row
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", required=True, help="input .surf.gii (smoothed, from stage 01)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cell-size", type=float, default=C.DEFAULT_CELL_SIZE,
                    help=f"cell size h in mm (default {C.DEFAULT_CELL_SIZE}, "
                         f"the paper's default resolution)")
    ap.add_argument("--meshtool", default=None)
    ap.add_argument("--reproject", action="store_true",
                    help="S6: snap boundary nodes onto the input surface")
    ap.add_argument("--skip-gmsh", action="store_true", help="bypass S3")
    ap.add_argument("--skip-meshtool", action="store_true", help="bypass S4")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--name", default=None, help="output basename (default: from the surface)")
    args = ap.parse_args()

    C.banner("STAGE 02 — full CORTET, without post-mesh smoothing")

    if not os.path.exists(args.surface):
        C.fail(f"input surface not found: {args.surface}")

    meshtool = C.find_meshtool(args.meshtool)
    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = args.work_dir or os.path.join(args.out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    base = args.name or os.path.basename(args.surface).split(".")[0]
    out_mesh = os.path.join(args.out_dir, f"{base}_h{args.cell_size}.mesh")

    stages = ["S1 CGAL", "S2 CARP"]
    if not args.skip_gmsh:
        stages.append("S3 Gmsh")
    if not args.skip_meshtool:
        stages.append("S4 meshtool")
    stages.append("S5 solver-ready")
    if args.reproject:
        stages.append("S6 reproject")

    C.step(f"Surface   : {args.surface}")
    C.step(f"Cell size : h = {args.cell_size} mm")
    C.step(f"Stages    : {' -> '.join(stages)}")
    C.step(f"Output    : {out_mesh}")

    if not args.skip_meshtool:
        ok, detail = C.check_meshtool_works(meshtool, work_dir)
        if not ok:
            C.fail(C.meshtool_problem(meshtool, detail) +
                   " Run run/00_check_environment.py. Do not trust any number "
                   "produced in this state.")
        C.step(f"meshtool  : {meshtool}  ({detail})")

    cmd = [sys.executable,
           C.script_path("cortet", "generation", "generate_tetrahedral_mesh.py"),
           "-i", args.surface, "-o", out_mesh,
           "--cell-size", str(args.cell_size),
           "--meshtool-path", meshtool, "--work-dir", work_dir]
    if args.reproject:
        cmd.append("--reproject")
    if args.skip_gmsh:
        cmd.append("--skip-gmsh")
    if args.skip_meshtool:
        cmd.append("--skip-meshtool")

    C.run_subprocess(cmd, "Generating the tetrahedral mesh")

    if not os.path.exists(out_mesh):
        C.fail(f"the generation script exited 0 but wrote no mesh at {out_mesh}")

    # ---- measure ----
    C.banner("Measurement (tet_qmetric_volume, 0 = regular, 1 = degenerate)")
    q, n_nodes, n_tets = C.measure_mesh_file(
        out_mesh, meshtool, work_dir, per_element=True, tag="s02")
    C.report_quality("full pipeline", q, n_nodes, n_tets)

    if not q:
        C.fail(f"the mesh was written to {out_mesh} but could not be measured, "
               f"so it cannot be certified against q_max < "
               f"{C.SOLVER_READY_THRESHOLD}. Quality measurement needs meshtool; "
               f"install it and re-run, or measure the file separately.")

    passed = q["max"] < C.SOLVER_READY_THRESHOLD

    # ---- compare against the published numbers where we have them ----
    exp = expected_for(base, args.cell_size)
    if exp:
        C.banner("Against the published values for this subject and h")
        print(f"  {'':<10}{'measured':>14}{'published':>14}{'difference':>14}")
        for key, label, fmt in (("nodes", "nodes", ","), ("tets", "tets", ","),
                                ("qbar", "q_bar", ".4f"), ("qmax", "q_max", ".4f")):
            got = {"nodes": n_nodes, "tets": n_tets,
                   "qbar": q["mean"], "qmax": q["max"]}[key]
            want = float(exp[key])
            if fmt == ",":
                print(f"  {label:<10}{int(got):>14,}{int(want):>14,}"
                      f"{int(got) - int(want):>+14,}")
            else:
                print(f"  {label:<10}{got:>14.4f}{want:>14.4f}{got - want:>+14.4f}")
        if exp["status"] != "confirmed":
            C.warn(f"the published value for this cell is marked "
                   f"'{exp['status']}' — your run may be the confirmation")

        # If node and tet counts are BOTH off by close to the same factor, the
        # input file is wrong, not the code. A genuine difference in settings or
        # tool version perturbs the two counts differently; a different scan
        # session or hemisphere scales the whole mesh, so the ratio holds. That
        # is how a wrong-session subject was caught during the Table 2 work: a
        # steady factor that did not move when the resolution changed.
        r_nodes = n_nodes / float(exp["nodes"])
        r_tets = n_tets / float(exp["tets"])
        off = max(abs(r_nodes - 1), abs(r_tets - 1)) > 0.02
        consistent = abs(r_nodes - r_tets) < 0.03 * max(r_nodes, r_tets)
        if off and consistent:
            C.warn(f"nodes and tets are both off by about the same factor "
                   f"({r_nodes:.2f}x and {r_tets:.2f}x). A consistent ratio "
                   f"points at the INPUT FILE, not the code — check you have "
                   f"the right subject, session, hemisphere and smoothing "
                   f"level before debugging anything else.")
        elif off:
            C.warn(f"counts differ from published ({r_nodes:.2f}x nodes, "
                   f"{r_tets:.2f}x tets). The two ratios differ, which points "
                   f"at settings or a tool version rather than the input file.")

    # ---- verdict ----
    C.banner("VERDICT")
    if passed:
        print(f"  PASS — worst element {q['max']:.4f} < {C.SOLVER_READY_THRESHOLD}, "
              f"solver-ready.")
        if q.get("n_above") is not None:
            print(f"  {q['n_above']:,} elements above the threshold "
                  f"(expected 0 for the full pipeline).")
        print(f"\n  Next: run/06_export_braingrowth.py to prepare it for the solver,")
        print(f"  or run/05_postmesh_smooth_and_clean.py first if your solver")
        print(f"  advances a growing free surface.\n")
    else:
        print(f"  FAIL — worst element {q['max']:.4f} >= "
              f"{C.SOLVER_READY_THRESHOLD}. Not solver-ready.\n")
        if args.skip_meshtool:
            print("  You passed --skip-meshtool, so S4 did not run. That is the "
                  "stage that\n  removes the degenerate tail; this result is "
                  "expected without it.\n")
        else:
            print("  At the latest gestational ages, deep narrow sulci are the "
                  "binding\n  constraint. Reducing --cell-size restores the "
                  "target at the cost of a\n  larger mesh.\n")

    summary = os.path.join(args.out_dir, "stage02_summary.csv")
    exists = os.path.exists(summary)
    with open(summary, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["surface", "h_mm", "stages", "nodes", "tets",
                        "qbar", "qmax", "n_above_0.6", "solver_ready"])
        w.writerow([os.path.basename(args.surface), args.cell_size,
                    "+".join(s.split()[0] for s in stages), n_nodes, n_tets,
                    f"{q['mean']:.4f}", f"{q['max']:.4f}",
                    q.get("n_above", ""), "yes" if passed else "no"])
    C.step(f"Summary appended: {summary}")
    print()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
