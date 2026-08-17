#!/usr/bin/env python3
r"""STAGE 01 -- pre-mesh surface smoothing (paper Section 3.5).

Smooths the input cortical surface with a surface-preserving Laplacian filter,
defining the stress-free reference configuration a folding simulation needs.
Writes one surface per iteration count plus a metrics CSV, then reports which N
removed the most noise while staying inside both caps: area loss <= 6 per cent,
drift <= 0.25 mm.

--ga picks a starting band from gestational age; --N overrides it. Look at the
results before committing:  wb_view smoothing/*.surf.gii

    python3 run/01_smooth_surface.py --surface subject_left_white.surf.gii \
        --out-dir smoothing/ --ga 21.9
    python3 run/01_smooth_surface.py --surface subject_left_white.surf.gii \
        --out-dir smoothing/ --N 0 10 20 30 50

WHAT TO SEND BACK from a cluster run: stage01_smoothing.csv and the console log.
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


# Starting bands from the fetal cohort. Starting points, not rules — the whole
# reason this script writes a sweep is that you should look before committing.
GA_BANDS = [
    (0.0,  23.0, [10, 20, 30]),
    (23.0, 25.0, [8, 12, 20]),
    (25.0, 29.0, [4, 6, 12]),
    (29.0, 33.0, [2, 4, 10]),
    (33.0, 34.0, [0, 2, 8]),
    (34.0, 99.0, [0, 1, 3]),
]

AREA_LOSS_CAP = 6.0     # per cent
CHAMFER_CAP = 0.25      # mm


def bands_for_ga(ga):
    for lo, hi, ns in GA_BANDS:
        if lo <= ga < hi:
            return ns
    return GA_BANDS[-1][2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", required=True, help="input .surf.gii")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=None,
                    help="smoothing iteration counts to write (default: from --ga, "
                         "or 0 10 20 30 40)")
    ap.add_argument("--ga", type=float, default=None,
                    help="gestational age in weeks; picks a starting band for --N")
    ap.add_argument("--angle", type=float, default=60.0,
                    help="dihedral-angle threshold in degrees. Inert above ~20 on "
                         "cortical surfaces; leave at 60 and vary N instead")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    C.banner("STAGE 01 — pre-mesh surface smoothing")

    if not os.path.exists(args.surface):
        C.fail(f"input surface not found: {args.surface}")

    if args.N is not None:
        n_values = args.N
        source = "given on the command line"
    elif args.ga is not None:
        n_values = bands_for_ga(args.ga)
        source = f"chosen from the GA {args.ga:.2f} wk band"
    else:
        n_values = [0, 10, 20, 30, 40]
        source = "default sweep (no --N or --ga given)"

    if 0 not in n_values:
        # N=0 is the unsmoothed copy; without it there is no baseline to compare
        # the metrics against, and every percentage below becomes meaningless.
        n_values = [0] + list(n_values)

    C.step(f"Surface   : {args.surface}")
    C.step(f"Output    : {args.out_dir}")
    C.step(f"N values  : {' '.join(map(str, n_values))}   ({source})")
    C.step(f"Angle     : {args.angle} deg")

    os.makedirs(args.out_dir, exist_ok=True)

    cmd = [sys.executable, C.script_path("cortet", "surface", "smooth_one_multi_N.py"),
           "--surface", args.surface, "--out-dir", args.out_dir,
           "--N", *[str(n) for n in n_values], "--angle", str(args.angle)]
    if args.overwrite:
        cmd.append("--overwrite")
    C.run_subprocess(cmd, "Running the surface-preserving smoothing sweep")

    # ---- read back the metrics and turn them into a recommendation ----
    metrics = sorted(glob.glob(os.path.join(args.out_dir, "*_smoothing_metrics.csv")))
    if not metrics:
        C.warn("no metrics CSV was written; cannot summarise")
        return 0

    rows = list(csv.DictReader(open(metrics[-1])))
    C.banner("Metrics")
    print(f"  {'N':>5}  {'noise removed %':>16}  {'area change %':>14}  "
          f"{'drift mm':>9}   within caps?")
    acceptable = []
    for r in rows:
        n = int(r["N"])
        noise = float(r["noise_removed_pct"])
        area = float(r["area_pct"])
        chamfer = float(r["chamfer_mm"])
        ok = abs(area) <= AREA_LOSS_CAP and chamfer <= CHAMFER_CAP
        if n > 0 and ok and noise > 0:
            acceptable.append((noise, n, area, chamfer))
        flag = "yes" if ok else ("NO — area" if abs(area) > AREA_LOSS_CAP
                                 else "NO — drift")
        if n == 0:
            flag = "baseline"
        print(f"  {n:>5}  {noise:>16.2f}  {area:>14.3f}  {chamfer:>9.4f}   {flag}")

    C.banner("Recommendation")
    if acceptable:
        noise, n, area, chamfer = max(acceptable)
        print(f"  N = {n} removes the most noise ({noise:.1f}%) while staying inside "
              f"both caps\n  (area {area:+.2f}% vs the {AREA_LOSS_CAP}% cap, drift "
              f"{chamfer:.3f} mm vs the {CHAMFER_CAP} mm cap).\n")
        print(f"  This is a starting point, not a verdict. Open them together and\n"
              f"  look before committing:   wb_view {args.out_dir}/*.surf.gii\n")
        # Locate the file that was ACTUALLY written rather than reconstructing
        # the name: the smoother derives its output stem from the input, and
        # that derivation is not a plain suffix swap (it drops a trailing
        # "_white", for one). Guessing the name here printed a path that did
        # not exist.
        written = glob.glob(os.path.join(args.out_dir, f"*_sp{n}.surf.gii"))
        chosen = written[0] if written else None
        if chosen:
            print(f"  Feed the one you choose into stage 02:\n"
                  f"    python3 run/02_generate_mesh.py --surface {chosen} ...\n")
        else:
            print(f"  Feed the N = {n} surface in {args.out_dir} into stage 02.\n")
    else:
        print("  No tested N removed noise while staying inside both caps.\n"
              "  For a well-folded surface that is a legitimate result: it may "
              "need little or no\n  smoothing. Try smaller N, and check the "
              "surface is what you think it is.\n")
        n = 0

    summary = os.path.join(args.out_dir, "stage01_summary.csv")
    with open(summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["surface", "angle_deg", "N_tested", "N_recommended",
                    "area_cap_pct", "chamfer_cap_mm"])
        w.writerow([os.path.basename(args.surface), args.angle,
                    " ".join(map(str, n_values)), n, AREA_LOSS_CAP, CHAMFER_CAP])
    C.step(f"Summary written: {summary}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
