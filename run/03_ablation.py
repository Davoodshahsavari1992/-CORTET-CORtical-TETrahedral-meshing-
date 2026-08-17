#!/usr/bin/env python3
r"""STAGE 03 -- stage ablation (the paper's Table 1).

Meshes the same surface three times -- CGAL only, + Gmsh, full -- and measures
all three identically. This is the paper's central argument in one command: the
mean barely moves (0.135 -> 0.129) while the count of solver-breaking elements
collapses from ~1.7e5 to zero. A mesh can have excellent mean quality and be
unusable, and the mean will not tell you.

Reporting that count needs meshtool's per-element dump; the summary line alone
cannot produce it. Three full meshing runs per subject, so start with one young
subject at the default resolution. Meshes are measured and deleted as it goes,
so disk never holds more than one.

    python3 run/03_ablation.py --surface subject_sp30.surf.gii \
        --out-dir ablation/ --meshtool ~/meshtool/meshtool
    python3 run/03_ablation.py --surface ... --out-dir ... --with-tetgen

WHAT TO SEND BACK from a cluster run: stage03_ablation.csv and the console log.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


CONFIGURATIONS = [
    ("CGAL only",   ["--skip-gmsh", "--skip-meshtool"], "S1-S2"),
    ("+ Gmsh",      ["--skip-meshtool"],                "S3"),
    ("Full CORTET", [],                                 "S4"),
]

# Table 1 of the paper: COHORT statistics over N = 194 subjects at h = 0.6 --
# mean +/- s.d. for q_bar, and median [min-max] ACROSS SUBJECTS for q_max and the
# threshold-exceeding count.
#
# THESE ARE NOT PER-SUBJECT EXPECTATIONS. Showing that one subject's q_max falls
# inside a 194-subject min-max envelope is a weak check: the envelope is wide by
# construction, and almost any plausible value lands in it. They are printed here
# as context only.
#
# The per-subject comparison is Table 2 (and results/table2_full_matrix.csv),
# which gives exact values for three named subjects at three resolutions -- but
# only for the FULL pipeline. There is no published per-subject value for the
# CGAL-only or +Gmsh configurations, so for those two rows the cohort envelope is
# genuinely all there is, and this script says so rather than implying otherwise.
PUBLISHED = {
    "CGAL only":   dict(qbar=0.135, qmax=0.780, qmax_lo=0.746, qmax_hi=0.820,
                        over=888, over_lo=274, over_hi=2449),
    "+ Gmsh":      dict(qbar=0.129, qmax=0.732, qmax_lo=0.685, qmax_hi=0.802,
                        over=191, over_lo=55, over_hi=519),
    "Full CORTET": dict(qbar=0.129, qmax=0.508, qmax_lo=0.462, qmax_hi=0.561,
                        over=0, over_lo=0, over_hi=0),
    "TetGen":      dict(qbar=0.242, qmax=0.996, qmax_lo=0.974, qmax_hi=1.000,
                        over=15955, over_lo=12095, over_hi=35414),
}


def run_configuration(surface, flags, cell_size, meshtool, work_dir):
    """Mesh once with the given flags, measure, delete. Returns (q, nodes, tets)."""
    mesh = os.path.join(work_dir, "_ablation_tmp.mesh")
    if os.path.exists(mesh):
        os.remove(mesh)

    cmd = [sys.executable,
           C.script_path("cortet", "generation", "generate_tetrahedral_mesh.py"),
           "-i", surface, "-o", mesh, "--cell-size", str(cell_size),
           "--meshtool-path", meshtool, "--work-dir", work_dir] + flags
    C.run_subprocess(cmd, f"Meshing with: {' '.join(flags) or '(no skips — full pipeline)'}")

    if not os.path.exists(mesh):
        C.warn("no mesh was written for this configuration")
        return None, None, None

    q, n_nodes, n_tets = C.measure_mesh_file(
        mesh, meshtool, work_dir, per_element=True, tag="s03")
    os.remove(mesh)
    return q, n_nodes, n_tets


def in_range(value, lo, hi):
    """Cohort-envelope check. Weak by nature -- see the note on PUBLISHED."""
    return "" if value is None else ("  within cohort range" if lo <= value <= hi
                                     else "  OUTSIDE cohort range")


def subject_expected(surface, cell_size):
    """The exact published value for this subject and h, if there is one.

    Table 2 / results/table2_full_matrix.csv are subject-level, so this is the
    comparison that actually tests a reproduction. It exists only for the FULL
    pipeline configuration.
    """
    path = C.script_path("results", "table2_h_resolution_study.csv")
    if not os.path.exists(path):
        return None
    base = os.path.basename(surface)
    for row in csv.DictReader(open(path)):
        if row["subject"] in base and abs(float(row["h_mm"]) - cell_size) < 1e-9:
            return row
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cell-size", type=float, default=C.DEFAULT_CELL_SIZE)
    ap.add_argument("--meshtool", default=None)
    ap.add_argument("--with-tetgen", action="store_true",
                    help="also run the TetGen baseline (stage 04) on each surface")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    C.banner("STAGE 03 — stage ablation")

    meshtool = C.find_meshtool(args.meshtool)
    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = args.work_dir or os.path.join(args.out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    ok, detail = C.check_meshtool_works(meshtool, work_dir)
    if not ok:
        C.fail(C.meshtool_problem(meshtool, detail) +
               " Run run/00_check_environment.py first. Every number this stage "
               "produces would be wrong.")
    C.step(f"meshtool  : {meshtool}  ({detail})")
    C.step(f"Cell size : h = {args.cell_size} mm")
    C.step(f"Subjects  : {len(args.surface)}")
    C.step(f"Runs      : {len(args.surface) * (len(CONFIGURATIONS) + bool(args.with_tetgen))} "
           f"meshing runs in total")

    out_csv = os.path.join(args.out_dir, "stage03_ablation.csv")
    write_header = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["surface", "h_mm", "configuration", "stages", "nodes",
                         "tets", "qbar", "qmax", "n_above_0.6"])

    for surface in args.surface:
        if not os.path.exists(surface):
            C.warn(f"skipping, not found: {surface}")
            continue

        C.banner(f"SUBJECT: {os.path.basename(surface)}", char="#")
        collected = []

        for label, flags, stages in CONFIGURATIONS:
            q, n_nodes, n_tets = run_configuration(
                surface, flags, args.cell_size, meshtool, work_dir)
            if not q:
                writer.writerow([os.path.basename(surface), args.cell_size, label,
                                 stages, "FAIL", "FAIL", "", "", ""])
                fh.flush()
                continue
            collected.append((label, q, n_nodes, n_tets))
            writer.writerow([os.path.basename(surface), args.cell_size, label, stages,
                             n_nodes, n_tets, f"{q['mean']:.4f}", f"{q['max']:.4f}",
                             q.get("n_above", "")])
            fh.flush()

        if args.with_tetgen:
            C.banner("TetGen baseline (stage 04, run inline)")
            rc = C.run_subprocess(
                [sys.executable, C.script_path("run", "04_tetgen_baseline.py"),
                 "--surface", surface, "--out-dir", args.out_dir,
                 "--meshtool", meshtool, "--work-dir", work_dir],
                "TetGen out-of-the-box on the same surface", allow_fail=True)
            if not rc:
                C.warn("TetGen baseline failed; see above. The ablation rows "
                       "are unaffected.")

        # ---- the comparison table for this subject ----
        C.banner(f"ABLATION — {os.path.basename(surface)}")
        print(f"  {'configuration':<14}{'q_bar':>9}{'q_max':>9}"
              f"{'elements q>0.6':>17}     vs Table 1 cohort envelope")
        for label, q, n_nodes, n_tets in collected:
            pub = PUBLISHED[label]
            over = q.get("n_above")
            print(f"  {label:<14}{q['mean']:>9.4f}{q['max']:>9.4f}"
                  f"{(f'{over:,}' if over is not None else '-'):>17}"
                  f"{in_range(q['max'], pub['qmax_lo'], pub['qmax_hi'])}")

        print("\n  Table 1 is a 194-subject envelope, so the column above is context,"
              "\n  not a reproduction test. The per-subject comparison follows.")

        # ---- the comparison that actually tests a reproduction ----
        exp = subject_expected(surface, args.cell_size)
        full = next((c for c in collected if c[0] == "Full CORTET"), None)
        if exp and full:
            _, q, n_nodes, n_tets = full
            C.banner(f"FULL PIPELINE vs Table 2, this subject at h = {args.cell_size}")
            print(f"  {'':<10}{'measured':>14}{'published':>14}{'difference':>14}")
            for key, lbl in (("nodes", "nodes"), ("tets", "tets")):
                got = n_nodes if key == "nodes" else n_tets
                want = int(exp[key])
                print(f"  {lbl:<10}{got:>14,}{want:>14,}{got - want:>+14,}")
            for key, lbl, got in (("qbar", "q_bar", q["mean"]),
                                  ("qmax", "q_max", q["max"])):
                want = float(exp[key])
                print(f"  {lbl:<10}{got:>14.4f}{want:>14.4f}{got - want:>+14.4f}")
            if exp["status"] != "confirmed":
                C.warn(f"the published value for this cell is marked "
                       f"'{exp['status']}'")
        elif full:
            print(f"\n  No published per-subject value for this subject at "
                  f"h = {args.cell_size},\n  so only the cohort envelope above is "
                  f"available. Table 2 covers\n  subject_GA21.86_left, subject_GA28.00_left "
                  f"and subject_GA33.86_left at h = 0.4 / 0.6 / 0.8.")

        if len(collected) == len(CONFIGURATIONS):
            first, last = collected[0][1], collected[-1][1]
            d_mean = abs(last["mean"] - first["mean"])
            print(f"\n  Mean quality moved by {d_mean:.4f} across the three "
                  f"configurations.")
            if first.get("n_above") is not None and last.get("n_above") is not None:
                print(f"  Elements above the solver-ready threshold went "
                      f"{first['n_above']:,} -> {last['n_above']:,}.")
                if last["n_above"] == 0 and first["n_above"] > 0:
                    print(f"\n  That is the paper's result: a mean that barely "
                          f"moves, and a tail that\n  vanishes. The mean could "
                          f"not have told you the first mesh was unusable.")
                elif last["n_above"] > 0:
                    print(f"\n  The full pipeline still leaves "
                          f"{last['n_above']:,} element(s) above threshold on "
                          f"this subject.\n  Reducing --cell-size is the lever.")

    fh.close()
    C.banner("DONE")
    print(f"  Results: {out_csv}")
    print(f"  Send that file back; it is the transferable artefact.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
