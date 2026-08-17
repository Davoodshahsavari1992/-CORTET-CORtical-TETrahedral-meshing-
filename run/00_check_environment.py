#!/usr/bin/env python3
r"""STAGE 00 -- check the environment before running anything else.

Run this first on any new machine, and again whenever a run behaves oddly. It
reports every dependency and marks which stage needs it, so a missing optional
package does not read as a broken install. Then it measures a single regular
tetrahedron, whose quality must come back near 0.

That last check is the point: meshtool can exit 0 and emit empty or wrong
output outside the correct module environment, with nothing pointing at the
cause. Verifying the binary exists is not enough. Exit 0 means every required
item passed.

    python3 run/00_check_environment.py
    python3 run/00_check_environment.py --meshtool ~/meshtool/meshtool
    python3 run/00_check_environment.py --quiet          # exit code only
"""

import argparse
import importlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from cortet import common as C  # noqa: E402


# (module, pip name, required?, what needs it)
PYTHON_DEPS = [
    ("numpy",     "numpy",     True,  "everything"),
    ("scipy",     "scipy",     True,  "field mapping, smoothing metrics"),
    ("nibabel",   "nibabel",   True,  "reading GIfTI surfaces"),
    ("trimesh",   "trimesh",   True,  "surface repair, nearest-surface queries"),
    ("meshio",    "meshio",    True,  "format conversion, stage 06 export"),
    ("pygalmesh", "pygalmesh", True,  "stage 02/03: CGAL tetrahedralisation"),
    ("gmsh",      "gmsh",      False, "stage 02/03: S3 Relocate3D smoothing"),
    ("pymeshlab", "pymeshlab", False, "stage 01/05: surface-preserving smoothing"),
    ("tetgen",    "tetgen",    False, "stage 04: TetGen baseline"),
    ("pyvista",   "pyvista",   False, "stage 04: TetGen baseline; rendering"),
]

STAGE_SCRIPTS = [
    "00_check_environment.py",
    "01_smooth_surface.py",
    "02_generate_mesh.py",
    "03_ablation.py",
    "04_tetgen_baseline.py",
    "05_postmesh_smooth_and_clean.py",
    "06_export_braingrowth.py",
]

CORE_SCRIPTS = [
    ("cortet", "common.py"),
    ("cortet", "surface", "smooth_one_multi_N.py"),
    ("cortet", "generation", "generate_tetrahedral_mesh.py"),
    ("cortet", "fields", "map_surface_fields_to_mesh.py"),
    ("cortet", "export", "export_mesh_formats.py"),
    ("cortet", "postmesh", "run_full_postmesh_pipeline.py"),
    ("cortet", "postmesh", "finalize_for_braingrowth.py"),
    ("cortet", "verify", "check_node_reordering.py"),
    ("cortet", "verify", "check_boundary_displacement.py"),
    ("cortet", "verify", "compare_h_direct.py"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshtool", default=None, help="path to the meshtool binary")
    ap.add_argument("--wb-command", default="wb_command")
    ap.add_argument("--quiet", action="store_true", help="suppress the report")
    args = ap.parse_args()

    out = (lambda *a, **k: None) if args.quiet else print
    problems = []
    optional_missing = []

    if not args.quiet:
        C.banner("STAGE 00 — environment check")

    # ---- Python ----------------------------------------------------------
    out(f"\n  Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 8):
        problems.append(f"Python {sys.version.split()[0]} is too old; need 3.8+")

    out("\n  Python packages")
    for mod, pip_name, required, used_by in PYTHON_DEPS:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            out(f"    {'OK  ':<5}{mod:<12} {version:<12} {used_by}")
        except Exception as exc:                              # noqa: BLE001
            kind = "MISS" if required else "opt "
            detail = type(exc).__name__ if not isinstance(exc, ImportError) else ""
            out(f"    {kind:<5}{mod:<12} {'-':<12} {used_by}"
                + (f"   [{detail}]" if detail else ""))
            if required:
                problems.append(f"{mod} is required ({used_by}); pip install {pip_name}")
            else:
                optional_missing.append((mod, used_by))

    # ---- external binaries ----------------------------------------------
    out("\n  External tools")

    meshtool = C.find_meshtool(args.meshtool)
    wb = shutil.which(args.wb_command)
    out(f"    {'OK  ' if os.path.exists(meshtool) else 'MISS':<5}"
        f"{'meshtool':<12} {meshtool}")
    if not os.path.exists(meshtool):
        problems.append(
            "meshtool not found. It is not pip-installable; see the README's "
            "Installation section. Override the location with --meshtool.")
    out(f"    {'OK  ' if wb else 'opt ':<5}{'wb_command':<12} {wb or '- (needed for --method wb and surface resampling)'}")
    if not wb:
        optional_missing.append(("wb_command", "stage 05 --method wb"))

    # ---- the check that actually matters --------------------------------
    out("\n  Does meshtool compute correct quality?")
    if os.path.exists(meshtool):
        with tempfile.TemporaryDirectory() as d:
            ok, detail = C.check_meshtool_works(meshtool, d)
        out(f"    {'OK  ' if ok else 'FAIL':<5}{detail}")
        if not ok:
            problems.append(
                f"meshtool runs but does not measure correctly ({detail}). This "
                f"is almost always a missing module environment, NOT a broken "
                f"mesh. Load the modules in docs/ENVIRONMENT.md and re-run. Do "
                f"not trust any quality number produced in this state.")
    else:
        out("    SKIP  (meshtool not found)")

    # ---- repository layout ----------------------------------------------
    out("\n  Repository layout")
    missing_files = []
    for parts in CORE_SCRIPTS:
        p = C.script_path(*parts)
        if not os.path.exists(p):
            missing_files.append(os.path.join(*parts))
    for name in STAGE_SCRIPTS:
        p = C.script_path("run", name)
        if not os.path.exists(p):
            missing_files.append(os.path.join("run", name))
    if missing_files:
        for m in missing_files:
            out(f"    MISS {m}")
        problems.append(f"{len(missing_files)} expected file(s) missing — the "
                        f"package is incomplete or was extracted partially")
    else:
        out(f"    OK   all {len(CORE_SCRIPTS) + len(STAGE_SCRIPTS)} expected "
            f"scripts present")

    for name, sub in (("results", "cohort_quality_194.csv"),
                      ("results", "table2_h_resolution_study.csv")):
        p = C.script_path(name, sub)
        out(f"    {'OK  ' if os.path.exists(p) else 'opt ':<5}{name}/{sub}")

    # ---- verdict ---------------------------------------------------------
    if not args.quiet:
        C.banner("VERDICT")
        if optional_missing:
            print("  Optional pieces absent — these stages are unavailable, "
                  "everything else works:")
            for mod, used_by in optional_missing:
                print(f"    - {mod}: {used_by}")
            print()
        if problems:
            print(f"  {len(problems)} problem(s) must be fixed before running "
                  f"the pipeline:\n")
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}\n")
            print("  Exit code 1.\n")
        else:
            print("  Environment is good. Proceed to run/01_smooth_surface.py.\n")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
