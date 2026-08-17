#!/usr/bin/env python3
r"""STAGE 05 -- post-mesh boundary smoothing, with the re-cleaning it requires.

Run this only if your solver advances a growing free surface, where the folds
arise from an instability of that surface so boundary faceting perturbs the very
surface being computed. Most targets do not need it, and for them stage 02's
output is already solver-ready.

Two rules are enforced, not merely documented: it always re-cleans after
smoothing, and it refuses a third cleaning pass at 0.3, which was tested and
does not converge.

    --method surfacepreserving   default; pymeshlab, in-process
    --method wb                  Connectome Workbench; reproduces the published
                                 Section 4.4 figures exactly

    python3 run/05_postmesh_smooth_and_clean.py \
        --mesh meshes/subject_h0.6.mesh --gii smoothing/subject_sp30.surf.gii \
        --out-dir postmesh/ --meshtool ~/meshtool/meshtool

WHAT TO SEND BACK from a cluster run: stage05_postmesh.csv, the displacement CSV
and the console log.
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


PUBLISHED = {
    "roughness_before": 0.191, "roughness_after": 0.134,
    "qmax_before": 0.488, "qmax_after": 0.500, "qmax_smoothed_only": 0.921,
    "disp_mean": 0.175, "disp_max": 0.480, "h": 0.4, "method": "wb",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="a mesh from stage 02 (skips regeneration)")
    ap.add_argument("--gii", required=True,
                    help="the input surface the mesh came from; needed for the "
                         "roughness and displacement checks")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", choices=["surfacepreserving", "wb"],
                    default="surfacepreserving")
    ap.add_argument("--cell-size", type=float, default=C.DEFAULT_CELL_SIZE,
                    help="only used if --mesh is omitted and the mesh is generated here")
    ap.add_argument("--meshtool", default=None)
    ap.add_argument("--wb-command", default="wb_command")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--interior-shells", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=1,
                    help="repeat the whole smooth-then-clean cycle this many "
                         "times. The smoother backs off when the result would be "
                         "invalid, and cleaning restores validity, so repeating "
                         "gets further than one pass while q_max stays pinned at "
                         "~0.50. Measured on one subject at h=0.6: roughness "
                         "0.01849 (1 cycle) -> 0.01700 (3) -> 0.01648 (5), zero "
                         "elements above threshold throughout. Returns diminish "
                         "sharply after 3; each cycle costs a full smooth+clean")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=C.POSTMESH_THRESHOLDS,
                    help="cleaning thresholds after smoothing. Default 0.7 0.5. "
                         "Adding 0.3 is a known failure; see this script's docstring")
    args = ap.parse_args()

    C.banner("STAGE 05 — post-mesh boundary smoothing and re-cleaning")

    if not os.path.exists(args.gii):
        C.fail(f"input surface not found: {args.gii}")
    if args.mesh and not os.path.exists(args.mesh):
        C.fail(f"mesh not found: {args.mesh}")

    meshtool = C.find_meshtool(args.meshtool)
    os.makedirs(args.out_dir, exist_ok=True)

    ok, detail = C.check_meshtool_works(meshtool, args.out_dir)
    if not ok:
        C.fail(C.meshtool_problem(meshtool, detail) +
               " Run run/00_check_environment.py first.")
    C.step(f"meshtool  : {meshtool}  ({detail})")

    # ---- guard rule 2 ----
    # It is not 0.3 specifically -- ANY third pass tighter than 0.5 is
    # destructive. Measured: 0.45 -> q_max 0.8826 with 134 elements above
    # threshold; 0.40 -> 0.8825 with 212; 0.30 -> 0.8823 with 210. The
    # original note named only 0.3 because that is the value that was tried.
    if len(args.thresholds) > 2 and min(args.thresholds) < 0.5 - 1e-9:
        C.banner("REFUSING TO RUN")
        print(f"  You asked for a third cleaning pass tighter than 0.5 "
              f"(thresholds {args.thresholds}).\n"
              f"  Every value tested there is destructive after boundary "
              f"smoothing:\n\n"
              f"      third pass at 0.45  ->  q_max 0.8826, 134 elements above 0.6\n"
              f"      third pass at 0.40  ->  q_max 0.8825, 212 elements above 0.6\n"
              f"      third pass at 0.30  ->  q_max 0.8823, 210 elements above 0.6\n\n"
              f"  If you want a smoother boundary, use --cycles instead. Repeating\n"
              f"  the whole smooth-and-clean cycle keeps q_max at ~0.50 with zero\n"
              f"  elements above threshold, and still improves roughness.\n")
        print("  If you want to reproduce that failure deliberately, edit the\n"
              "  thresholds in cortet/postmesh/run_full_postmesh_pipeline.py and\n"
              "  call it directly — but do not ship a mesh made this way.\n")
        print("\n")
        return 1

    # ---- record what we are starting from ----
    before = None
    if args.mesh:
        C.banner("Before smoothing")
        before, n_nodes, n_tets = C.measure_mesh_file(
            args.mesh, meshtool, args.out_dir, per_element=True, tag="s05pre")
        C.report_quality("raw mesh from stage 02", before, n_nodes, n_tets)
        if before and before["max"] >= C.SOLVER_READY_THRESHOLD:
            C.warn(f"the input mesh is already above the solver-ready threshold "
                   f"({before['max']:.4f}). Smoothing will not fix that; fix the "
                   f"mesh first.")

    C.step(f"Method    : {args.method}")
    C.step(f"Thresholds: {' '.join(str(t) for t in args.thresholds)}  "
           f"(2 passes, as required)")
    if args.method != PUBLISHED["method"]:
        C.step(f"Note      : the published Section 4.4 figures were measured with "
               f"--method {PUBLISHED['method']}")

    if args.cycles < 1:
        C.fail("--cycles must be at least 1")
    if args.cycles > 1:
        C.step(f"Cycles    : {args.cycles} (smooth + clean, repeated)")

    current_mesh = args.mesh
    for cyc in range(args.cycles):
        cyc_dir = (args.out_dir if args.cycles == 1
                   else os.path.join(args.out_dir, f"cycle{cyc + 1}"))
        os.makedirs(cyc_dir, exist_ok=True)

        cmd = [sys.executable,
               C.script_path("cortet", "postmesh", "run_full_postmesh_pipeline.py"),
               "--gii", args.gii, "--out-dir", cyc_dir,
               "--method", args.method, "--meshtool", meshtool,
               "--iterations", str(args.iterations),
               "--interior-shells", str(args.interior_shells),
               "--thresholds", *[str(t) for t in args.thresholds]]
        if current_mesh:
            cmd += ["--mesh", current_mesh]
        else:
            cmd += ["--cell-size", str(args.cell_size)]
        if args.method == "wb":
            cmd += ["--wb-command", args.wb_command]

        label = ("Smoothing, re-cleaning, finalising and verifying"
                 if args.cycles == 1
                 else f"Cycle {cyc + 1}/{args.cycles}: smooth, re-clean, finalise, verify")
        C.run_subprocess(cmd, label)

        # feed this cycle's finished mesh into the next
        produced = sorted(glob.glob(os.path.join(cyc_dir, "*_FINAL.mesh")))
        if not produced:
            produced = sorted(glob.glob(os.path.join(cyc_dir, "*BRAINGROWTH_READY.mesh")))
        if not produced:
            C.fail(f"cycle {cyc + 1} produced no mesh in {cyc_dir}")
        current_mesh = produced[-1]

        if args.cycles > 1:
            q_c = C.measure_mesh_file(current_mesh, meshtool, cyc_dir, tag=f"c{cyc}")[0]
            if q_c:
                C.step(f"after cycle {cyc + 1}: q_max {q_c['max']:.4f}")
                if q_c["max"] >= C.SOLVER_READY_THRESHOLD:
                    C.warn(f"q_max reached {q_c['max']:.4f} at cycle {cyc + 1}; "
                           f"stopping rather than cycling further")
                    break

    # ---- measure the result ----
    # With --cycles the loop already tracked the mesh forward, and the last one
    # it produced IS the result. Only fall back to searching --out-dir for the
    # single-cycle case, where nothing was tracked.
    if args.cycles > 1 and current_mesh:
        final = current_mesh
    else:
        base = os.path.splitext(os.path.basename(args.mesh or args.gii))[0]
        candidates = [
            os.path.join(args.out_dir, f"{base}_BRAINGROWTH_READY.mesh"),
            os.path.join(args.out_dir, f"{base}_final.mesh"),
        ]
        final = next((c for c in candidates if os.path.exists(c)), None)
        if final is None:
            found = sorted(glob.glob(os.path.join(args.out_dir, "*BRAINGROWTH_READY.mesh")))
            final = found[-1] if found else None

    after = None
    if final:
        C.banner("After smoothing + 2-pass re-cleaning")
        after, n_nodes, n_tets = C.measure_mesh_file(
            final, meshtool, args.out_dir, per_element=True, tag="s05post")
        C.report_quality("final mesh", after, n_nodes, n_tets)
    else:
        C.warn("could not locate the final mesh to measure it independently; "
               "the orchestrator's own report above still applies")

    # ---- compare against the published trade ----
    C.banner("The trade, against the published values")
    print(f"  Published, at h = {PUBLISHED['h']} with --method "
          f"{PUBLISHED['method']}, one subject:")
    print(f"    roughness  {PUBLISHED['roughness_before']:.3f} -> "
          f"{PUBLISHED['roughness_after']:.3f}   (about 30% removed)")
    print(f"    q_max      {PUBLISHED['qmax_before']:.3f} -> "
          f"{PUBLISHED['qmax_after']:.3f}   (still inside the criterion)")
    print(f"    boundary   {PUBLISHED['disp_mean']:.3f} mm mean, "
          f"{PUBLISHED['disp_max']:.3f} mm max from the input surface")
    if before and after:
        print(f"\n  This run:")
        print(f"    q_max      {before['max']:.3f} -> {after['max']:.3f}"
              f"   ({after['max'] - before['max']:+.3f})")
        if after["max"] < C.SOLVER_READY_THRESHOLD:
            print(f"\n  PASS — the mesh is still solver-ready after smoothing.")
        else:
            print(f"\n  FAIL — q_max {after['max']:.4f} is at or above "
                  f"{C.SOLVER_READY_THRESHOLD}. Do not simulate with this mesh.")
    print(f"\n  Roughness and boundary displacement are reported by the steps "
          f"above;\n  they are measured against {os.path.basename(args.gii)}.\n")

    summary = os.path.join(args.out_dir, "stage05_summary.csv")
    exists = os.path.exists(summary)
    with open(summary, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["mesh", "gii", "method", "thresholds",
                        "qmax_before", "qmax_after", "solver_ready_after"])
        w.writerow([os.path.basename(args.mesh or "(generated here)"),
                    os.path.basename(args.gii), args.method,
                    " ".join(str(t) for t in args.thresholds),
                    f"{before['max']:.4f}" if before else "",
                    f"{after['max']:.4f}" if after else "",
                    ("yes" if after and after["max"] < C.SOLVER_READY_THRESHOLD
                     else "no" if after else "")])
    C.step(f"Summary appended: {summary}")
    print(f"\n  Next: run/06_export_braingrowth.py on the final mesh.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
