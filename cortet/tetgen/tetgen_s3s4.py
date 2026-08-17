#!/usr/bin/env python3
r"""tetgen_s3s4.py -- run a TetGen mesh through CORTET's Stage 3 and Stage 4.

Applies the identical Gmsh Relocate3D smoothing and meshtool 3-pass cleaning
the CORTET pipeline uses, to a TetGen-generated CARP mesh. This answers
whether the worst-element guarantee comes from the optimisation stages or from
the CGAL tetrahedralisation itself.

    python3 tetgen_s3s4.py --carp-in tetgen_output/subject \
        --out-dir tetgen_S3S4/ --meshtool ~/meshtool/meshtool
    python3 tetgen_s3s4.py --selftest
"""
import argparse
import os
import subprocess
import time

import numpy as np


# ---------------------------------------------------------------------------
# CARP I/O -- verbatim from gii_to_braingrowth_optimized-9march.py
# (write_carp / read_carp), confirmed identical by direct comparison.
# ---------------------------------------------------------------------------
def write_carp(points, tets, basename):
    n_nodes = points.shape[0]
    n_tets = tets.shape[0]
    with open(f"{basename}.pts", "w") as f:
        f.write(f"{n_nodes}\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    with open(f"{basename}.elem", "w") as f:
        f.write(f"{n_tets}\n")
        for t in tets:
            f.write(f"Tt {t[0]} {t[1]} {t[2]} {t[3]} 1\n")
    with open(f"{basename}.lon", "w") as f:
        f.write("1\n")
        for _ in range(n_tets):
            f.write("1.000000 0.000000 0.000000\n")


def read_carp(basename):
    with open(f"{basename}.pts") as f:
        n_nodes = int(f.readline().strip())
        points = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            points[i] = [float(x) for x in f.readline().split()]
    with open(f"{basename}.elem") as f:
        n_tets = int(f.readline().strip())
        tets = np.zeros((n_tets, 4), dtype=int)
        for i in range(n_tets):
            parts = f.readline().split()
            tets[i] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
    return points, tets


# ---------------------------------------------------------------------------
# STAGE 3: Gmsh Relocate3D -- verbatim logic from stage3_gmsh_relocate() in
# the real generation script (same Mesa GLU path handling, same 5-pass
# Relocate3D loop, same meshio round-trip through .msh).
# ---------------------------------------------------------------------------
def stage3_gmsh_relocate(basename_in, basename_out, n_passes=5):
    print("=" * 70)
    print(f"STAGE 3: Gmsh Relocate3D optimization ({n_passes} passes)")
    print("=" * 70)

    # see generate_tetrahedral_mesh.py and docs/ENVIRONMENT.md
    gl_libs = os.environ.get("CORTET_GL_LIBS", "")
    if gl_libs:
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if gl_libs not in existing_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{gl_libs}:{existing_ld}"

    try:
        import gmsh
    except (ImportError, OSError) as e:
        print(f"  WARNING: Gmsh not available ({e})")
        print("  Install with: pip install gmsh --user")
        return False

    import meshio

    print("  Converting CARP -> Gmsh format...")
    points, tets = read_carp(basename_in)

    msh_in = f"{basename_in}.msh"
    msh_out = f"{basename_out}.msh"
    mesh_obj = meshio.Mesh(points=points, cells=[("tetra", tets)])
    meshio.write(msh_in, mesh_obj, file_format="gmsh22")

    print(f"  Running {n_passes} Relocate3D passes...")
    t0 = time.time()
    try:
        gmsh.initialize(["-noenv", "-nopopup"])
        gmsh.open(msh_in)
        for i in range(n_passes):
            gmsh.model.mesh.optimize("Relocate3D")
            print(f"    Pass {i+1}/{n_passes} done")
        gmsh.write(msh_out)
        gmsh.finalize()
    except Exception as e:
        print(f"  ERROR: Gmsh optimization failed: {e}")
        try:
            gmsh.finalize()
        except Exception:
            pass
        for f in [msh_in, msh_out]:
            if os.path.exists(f):
                os.remove(f)
        return False

    print(f"  Gmsh time: {time.time() - t0:.1f}s")

    print("  Converting Gmsh -> CARP format...")
    gmsh_mesh = meshio.read(msh_out)
    gmsh_pts = gmsh_mesh.points
    gmsh_tets = None
    for cell_block in gmsh_mesh.cells:
        if cell_block.type == "tetra":
            gmsh_tets = cell_block.data
            break
    if gmsh_tets is None:
        print("  ERROR: No tetrahedra in Gmsh output.")
        return False

    write_carp(gmsh_pts, gmsh_tets, basename_out)
    print(f"  Result: {gmsh_pts.shape[0]} nodes, {gmsh_tets.shape[0]} tets")

    for f in [msh_in, msh_out]:
        if os.path.exists(f):
            os.remove(f)
    return True


# ---------------------------------------------------------------------------
# STAGE 4: meshtool 3-pass clean -- verbatim from stage4_meshtool_clean() in
# the real generation script. NOTE: no -smth flag, matching the real
# script exactly (meshtool's own default, 0.00, applies) -- do not add one.
# ---------------------------------------------------------------------------
def stage4_meshtool_clean(basename_in, basename_out, meshtool_path, thresholds=None):
    print("=" * 70)
    print("STAGE 4: Iterative meshtool quality cleaning")
    print("=" * 70)

    if thresholds is None:
        thresholds = [0.7, 0.5, 0.3]

    current_input = basename_in
    for i, thr in enumerate(thresholds):
        pass_num = i + 1
        current_output = (f"{basename_out}_pass{pass_num}"
                          if pass_num < len(thresholds) else basename_out)

        print(f"\n  Pass {pass_num}/{len(thresholds)}: threshold={thr}")
        cmd = [meshtool_path, "clean", "quality",
               f"-msh={current_input}", "-ifmt=carp_txt",
               f"-outmsh={current_output}", "-ofmt=carp_txt",
               f"-thr={thr}"]
        # deliberately NO -smth flag here -- matches the real Stage 4 exactly

        result = subprocess.run(cmd, capture_output=True, text=True)
        for out_line in result.stdout.split("\n"):
            if "min:" in out_line.lower() and "max:" in out_line.lower():
                print(f"    {out_line.strip()}")
            if "Number of elements" in out_line:
                print(f"    {out_line.strip()}")

        if not os.path.exists(f"{current_output}.pts"):
            import shutil
            print("  No output written -- copying input (already clean at this threshold).")
            for ext in [".pts", ".elem", ".lon"]:
                src, dst = f"{current_input}{ext}", f"{current_output}{ext}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        current_input = current_output
    print()


# ---------------------------------------------------------------------------
def query_quality(basename, meshtool_path, label):
    """Uses -ifmt (INPUT format), not the bare -fmt in the brief -- meshtool
    has no -fmt flag; the brief's Run A/B commands both had this typo."""
    cmd = [meshtool_path, "query", "quality", f"-msh={basename}", "-ifmt=carp_txt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"\n--- quality: {label} ---")
    print(result.stdout)
    if result.returncode != 0:
        print(f"  (meshtool exited {result.returncode}; stderr: {result.stderr.strip()[:300]})")
    return result.stdout


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tetgen_base = args.carp_in
    query_quality(tetgen_base, args.meshtool, "TetGen output, before S3/S4")

    s3_base = os.path.join(args.out_dir, "after_s3")
    ok = stage3_gmsh_relocate(tetgen_base, s3_base, n_passes=args.gmsh_passes)
    if not ok:
        print("Stage 3 failed or was skipped -- see warnings above. "
              "Not proceeding to Stage 4 on a Gmsh output that doesn't exist.")
        return
    query_quality(s3_base, args.meshtool, "after Stage 3 (Gmsh)")

    s4_base = os.path.join(args.out_dir, "after_s4")
    stage4_meshtool_clean(s3_base, s4_base, args.meshtool, thresholds=args.thresholds)
    query_quality(s4_base, args.meshtool, "after Stage 4 (meshtool 3-pass, final)")


# ---------------------------------------------------------------------------
def selftest():
    """No Gmsh/meshtool needed -- verifies the CARP read/write round-trip
    used by every stage in this script, since a bug there would corrupt
    every downstream stage silently."""
    import tempfile
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)

    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "test")
        write_carp(pts, tets, base)
        for ext in (".pts", ".elem", ".lon"):
            assert os.path.exists(base + ext), f"missing {ext}"

        pts2, tets2 = read_carp(base)
        assert np.allclose(pts2, pts), "point round-trip failed"
        assert np.array_equal(tets2, tets), "tet round-trip failed"

    print("selftest PASS: CARP read/write round-trip is correct (verbatim")
    print("               logic from the real generation script). Gmsh and")
    print("               meshtool stages themselves need those tools")
    print("               installed -- run on CREATE and read the printed")
    print("               output at each stage.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carp-in", help="basename of the TetGen CARP mesh (no extension)")
    ap.add_argument("--out-dir", default="tetgen_S3S4")
    ap.add_argument("--meshtool", default=os.path.expanduser("~/meshtool/meshtool"))
    ap.add_argument("--gmsh-passes", type=int, default=5)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.5, 0.3],
                     help="matches the real generation script's Stage 4 default")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.carp_in:
        ap.error("--carp-in is required (or --selftest)")

    run(args)


if __name__ == "__main__":
    main()
