#!/usr/bin/env python3
r"""STAGE 06 -- export to the solver, with every convention applied AND verified.

Takes a mesh from stage 02 or stage 05 and hands back a file the solver will
accept: explicit boundary faces, surface-first node ordering, and the
per-element node ordering and signed-volume convention it reads. Then it
re-opens the written file and tests each property on the file itself, because
checking that the finaliser exited 0 is not the same as checking that its
output has the properties the solver depends on. Non-zero exit if any check
fails, so a batch script cannot walk on with a bad mesh.

It also normalises the incoming convention first. The finaliser is
convention-preserving: handed a mesh in standard positive orientation -- from
TetGen, from a plain Delaunay triangulation, from another group's mesher -- it
faithfully preserves that, giving correct faces, correct node ordering, and a
file still not in the convention the solver reads. Nothing errors.

    python3 run/06_export_braingrowth.py --mesh meshes/subject_h0.6.mesh \
        --out-dir solver_ready/ --meshtool ~/meshtool/meshtool

    # carry thickness / growth / labels onto the volume mesh
    python3 run/06_export_braingrowth.py --mesh ... --out-dir ... \
        --fields --gii subject_sp30.surf.gii --thickness subject_thickness.shape.gii

    # also write the mesh for a different solver
    python3 run/06_export_braingrowth.py --mesh ... --out-dir ... \
        --also-export solver_ready/mesh.inp

WHAT TO SEND BACK from a cluster run: stage06_export.csv and the console log.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


def normalise_input_orientation(mesh_in, work_path):
    """
    Put the input into the SOLVER's orientation convention before finalising,
    and report what convention it arrived in.

    WHY THIS IS NEEDED, and it is subtle. finalize_for_braingrowth.py is
    CONVENTION-PRESERVING: it undoes the n3<->n4 swap when it reads, rebuilds
    the face section and node ordering, then re-applies the swap when it
    writes. Its contract is a mesh that is already in this pipeline's .mesh
    convention, where signed volumes are NEGATIVE. Hand it a mesh in the
    standard positive convention and it will faithfully preserve that — giving
    you a file that has correct faces and correct node ordering and is still
    not in the convention the solver reads. Nothing errors.

    It also flips FACES for outward orientation but never TETS. Inside the
    normal pipeline that is fine, because the tets were made consistent
    upstream in mesh_to_carp_and_clean.py. A mesh from anywhere else — TetGen,
    a plain Delaunay triangulation, another group's mesher — carries no such
    guarantee, and per-element orientation from Delaunay is arbitrary. Applying
    the swap to a mixed-orientation mesh yields a file satisfying NEITHER
    convention, with roughly half the elements inverted for the solver.

    So: make every element consistent, and specifically negative, which is what
    finalize then preserves.

    Returns (path_to_use, n_changed, arrived_as).
    """
    points, tets = C.read_braingrowth_mesh(mesh_in)
    vols = C.signed_volumes(points, tets)
    n_neg, n_pos = int((vols < 0).sum()), int((vols > 0).sum())

    if int((vols == 0).sum()):
        arrived = "degenerate"
    elif n_pos == 0:
        arrived = "solver convention (all negative)"
    elif n_neg == 0:
        arrived = "standard convention (all positive)"
    else:
        arrived = f"MIXED ({n_neg:,} negative, {n_pos:,} positive)"

    # target: every element negative, which is what the solver's loader expects
    # once finalize has round-tripped the swap
    positive, _ = C.orient_positive(points, tets)
    target = positive.copy()
    target[:, [2, 3]] = target[:, [3, 2]]          # positive -> negative
    n_changed = int((target != tets).any(axis=1).sum())

    if n_changed == 0:
        return mesh_in, 0, arrived

    with open(work_path, "w") as fh:
        fh.write(f"{len(points)}\n")
        for p in points:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        fh.write(f"{len(target)}\n")
        for t in target:
            fh.write(f"   1  {t[0]+1:>8}  {t[1]+1:>8}  {t[2]+1:>8}  {t[3]+1:>8}\n")
    return work_path, n_changed, arrived


def verify_conventions(path):
    """
    Re-open a finalised .mesh and test all five conventions on the file itself.
    Returns a list of (name, passed, detail).
    """
    checks = []

    # ---- 1: five fields per tet line, three sections present ----
    try:
        points, tets, faces = C.read_braingrowth_mesh(path, want_faces=True)
        checks.append(("1. five integers per tet line", True,
                       f"{len(tets):,} tets parsed with 5 fields each"))
    except ValueError as exc:
        checks.append(("1. five integers per tet line", False, str(exc)))
        return checks
    except Exception as exc:                                  # noqa: BLE001
        checks.append(("1. five integers per tet line", False,
                       f"{type(exc).__name__}: {exc}"))
        return checks

    # ---- 5: boundary-face section ----
    if faces is None or len(faces) == 0:
        checks.append(("5. boundary-face section present", False,
                       "no third section — the solver has no contact detection "
                       "and no surface normals"))
    else:
        # every boundary face must belong to exactly one tet
        from collections import defaultdict
        counts = defaultdict(int)
        for t in tets:
            for combo in ((t[0], t[1], t[2]), (t[0], t[1], t[3]),
                          (t[0], t[2], t[3]), (t[1], t[2], t[3])):
                counts[tuple(sorted(combo))] += 1
        true_boundary = {k for k, v in counts.items() if v == 1}
        listed = {tuple(sorted(f)) for f in faces}
        if listed == true_boundary:
            checks.append(("5. boundary-face section present", True,
                           f"{len(faces):,} faces, exactly the true boundary"))
        else:
            checks.append(("5. boundary-face section present", False,
                           f"{len(faces):,} listed vs {len(true_boundary):,} "
                           f"actual boundary faces "
                           f"({len(listed - true_boundary):,} spurious, "
                           f"{len(true_boundary - listed):,} missing)"))
        surface_nodes = sorted({i for f in true_boundary for i in f})

        # ---- 4: surface nodes first, contiguously ----
        n_surface = len(surface_nodes)
        if surface_nodes == list(range(n_surface)):
            checks.append(("4. surface nodes ordered first", True,
                           f"{n_surface:,} surface nodes occupy indices "
                           f"0..{n_surface - 1}"))
        else:
            checks.append(("4. surface nodes ordered first", False,
                           f"{n_surface:,} surface nodes, but the highest index "
                           f"is {max(surface_nodes):,} — they are not contiguous "
                           f"at the start, so surface-indexed arrays will "
                           f"silently read wrong entries"))

    # ---- 3: signed volume negative in the solver convention ----
    vols = C.signed_volumes(points, tets)
    n_neg = int((vols < 0).sum())
    n_zero = int((vols == 0).sum())
    frac = n_neg / max(len(vols), 1)
    if n_zero:
        checks.append(("3. signed-volume convention", False,
                       f"{n_zero:,} element(s) have exactly zero volume — "
                       f"degenerate, whatever the convention"))
    elif frac > 0.999:
        checks.append(("3. signed-volume convention", True,
                       f"all {n_neg:,} tets negative, as the solver expects "
                       f"(a quality tool will call this invalid; that is correct)"))
    elif frac < 0.001:
        checks.append(("3. signed-volume convention", False,
                       f"all tets are POSITIVE. This file is in the standard "
                       f"convention, not the solver's — the n3/n4 swap did not "
                       f"happen"))
    else:
        checks.append(("3. signed-volume convention", False,
                       f"mixed orientation: {n_neg:,} negative, "
                       f"{len(vols) - n_neg:,} positive. Neither convention "
                       f"holds; some elements are inverted"))

    # ---- 2: the n3/n4 swap is what produces convention 3 ----
    unswapped = np.array(tets, copy=True)
    unswapped[:, [2, 3]] = unswapped[:, [3, 2]]
    n_pos_after = int((C.signed_volumes(points, unswapped) > 0).sum())
    if n_pos_after / max(len(vols), 1) > 0.999:
        checks.append(("2. n3<->n4 swap applied on write", True,
                       "undoing the swap yields a fully positive mesh, which is "
                       "the property the solver's loader relies on"))
    else:
        checks.append(("2. n3<->n4 swap applied on write", False,
                       f"undoing the swap leaves only {n_pos_after:,} of "
                       f"{len(vols):,} tets positive; the file is not in the "
                       f"expected pre-swapped form"))

    return checks


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="mesh from stage 02 or stage 05")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meshtool", default=None)
    ap.add_argument("--skip-finalize", action="store_true",
                    help="the mesh is already finalised; only verify it")
    ap.add_argument("--also-export", default=None,
                    help="additionally write this path in another FE format "
                         "(suffix decides: .inp .feb .xdmf .vtu .vtk .msh)")
    ap.add_argument("--fields", action="store_true",
                    help="also map per-vertex surface fields onto the volume")
    ap.add_argument("--gii", default=None, help="input surface, required by --fields")
    ap.add_argument("--thickness", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    C.banner("STAGE 06 — solver-ready export, with every convention verified")

    if not os.path.exists(args.mesh):
        C.fail(f"mesh not found: {args.mesh}")
    if args.fields and not args.gii:
        C.fail("--fields needs --gii (the surface the fields are defined on)")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.mesh))[0]
    final = os.path.join(args.out_dir, f"{base}_BRAINGROWTH_READY.mesh")

    # ---- apply the conventions ----
    if args.skip_finalize:
        C.step("--skip-finalize given; verifying the input mesh as-is")
        final = args.mesh
    else:
        # Normalise tet orientation FIRST. The finaliser flips faces but not
        # tets, and applying the n3<->n4 swap to a mixed-orientation mesh
        # produces a file that satisfies neither convention, silently.
        source, n_changed, arrived = normalise_input_orientation(
            args.mesh, os.path.join(args.out_dir, f"{base}_oriented.mesh"))
        C.step(f"Input arrived in: {arrived}")
        if n_changed:
            C.step(f"Reoriented {n_changed:,} tet(s) into the solver convention "
                   f"before finalising")
            C.step("(the finaliser preserves whatever convention it is given, "
                   "and flips faces but never tets)")
        else:
            C.step("Already in the solver convention; nothing to reorient")

        C.run_subprocess(
            [sys.executable,
             C.script_path("cortet", "postmesh", "finalize_for_braingrowth.py"),
             "--mesh", source, "--out", final],
            "Applying the solver conventions")
        if not os.path.exists(final):
            C.fail(f"the finaliser exited 0 but wrote nothing at {final}")

    # ---- verify them, on the written file ----
    C.banner("Verifying the five conventions, on the written file")
    checks = verify_conventions(final)
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print(f"        {detail}")
    all_passed = all(p for _, p, _ in checks)

    # ---- independent second opinion on ordering ----
    C.banner("Independent check: cortet/verify/check_node_reordering.py")
    rc = subprocess.run(
        [sys.executable, C.script_path("cortet", "verify", "check_node_reordering.py"),
         "--mesh", final]).returncode
    if rc != 0:
        C.warn(f"check_node_reordering.py exited {rc}")
        all_passed = False

    # ---- measure the file we are actually shipping ----
    meshtool = C.find_meshtool(args.meshtool)
    q = None
    if os.path.exists(meshtool):
        ok, detail = C.check_meshtool_works(meshtool, args.out_dir)
        if ok:
            C.banner("Quality of the finished file")
            q, n_nodes, n_tets = C.measure_mesh_file(
                final, meshtool, args.out_dir, per_element=True, tag="s06")
            C.report_quality("solver-ready mesh", q, n_nodes, n_tets)
            if q and q["max"] >= C.SOLVER_READY_THRESHOLD:
                C.warn(f"q_max {q['max']:.4f} is at or above "
                       f"{C.SOLVER_READY_THRESHOLD}; the conventions are right "
                       f"but the mesh is not solver-ready")
                all_passed = False
        else:
            C.warn(C.meshtool_problem(meshtool, detail) +
                   " Skipping the quality check.")
    else:
        C.warn("meshtool not found; conventions verified but quality not measured")

    # ---- optional: per-vertex fields ----
    if args.fields:
        C.banner("Mapping per-vertex surface fields onto the volume")
        print("  Refinement discarded the input vertices, so these fields cannot "
              "be carried\n  across by index. This maps them geometrically: "
              "barycentric for continuous\n  fields, nearest-vertex for discrete "
              "ones, interior nodes from the nearest\n  surface node.\n")
        cmd = [sys.executable,
               C.script_path("cortet", "fields", "map_surface_fields_to_mesh.py"),
               "--gii-surface", args.gii, "--mesh", final,
               "--output-dir", os.path.join(args.out_dir, "fields")]
        for flag, value in (("--thickness", args.thickness),
                            ("--labels", args.labels),
                            ("--parcellation", args.parcellation)):
            if value:
                cmd += [flag, value]
        C.run_subprocess(cmd, "Field transfer", allow_fail=True)

    # ---- optional: another FE format ----
    if args.also_export:
        C.banner(f"Also exporting to {args.also_export}")
        C.run_subprocess(
            [sys.executable, C.script_path("cortet", "export", "export_mesh_formats.py"),
             "--mesh", final, "--out", args.also_export],
            "Format export", allow_fail=True)

    # ---- record and verdict ----
    report = os.path.join(args.out_dir, "stage06_checks.csv")
    exists = os.path.exists(report)
    with open(report, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["mesh", "convention", "passed", "detail"])
        for name, passed, detail in checks:
            w.writerow([os.path.basename(final), name,
                        "yes" if passed else "no", detail])

    C.banner("VERDICT")
    if all_passed:
        print(f"  All five conventions verified on the written file, and the "
              f"mesh is\n  solver-ready.\n")
        print(f"  Solver-ready mesh:  {final}")
        if q:
            print(f"  Worst element:      {q['max']:.4f}  "
                  f"(criterion: < {C.SOLVER_READY_THRESHOLD})")
        print(f"  Checks:             {report}\n")
        print(f"  The solver is not part of this package. See the README's Scope\n"
              f"  section and https://github.com/rousseau/BrainGrowth\n")
        return 0

    failed = [n for n, p, _ in checks if not p]
    print(f"  {len(failed)} check(s) FAILED:")
    for n in failed:
        print(f"    - {n}")
    print(f"\n  Do NOT simulate with this mesh. Each of these fails silently at\n"
          f"  run time: the simulation would complete and look plausible while\n"
          f"  the mechanics underneath are wrong. That is the whole reason this\n"
          f"  stage verifies instead of trusting.\n")
    print(f"  Details: {report}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
