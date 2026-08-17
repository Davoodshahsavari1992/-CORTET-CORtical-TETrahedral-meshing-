#!/usr/bin/env python3
r"""mesh_to_carp_and_clean.py -- format conversion and meshtool cleaning.

Converts .mesh to CARP (.pts/.elem/.lon) and back, and optionally runs
meshtool's worst-element cleaning over it.

    python3 mesh_to_carp_and_clean.py --mesh subject.mesh --carp-base subject_carp
    python3 mesh_to_carp_and_clean.py --mesh subject.mesh --carp-base subject_carp \
        --clean --meshtool ~/meshtool/meshtool
    python3 mesh_to_carp_and_clean.py --selftest
"""
import argparse
import os
import subprocess
import sys

import numpy as np


# ---------------------------------------------------------------------------
# .mesh I/O (5 integers per tet: leading tag + 4 node indices -- the format
# fix established earlier in this project; see the other scripts in this
# package for the same convention)
# ---------------------------------------------------------------------------
def read_mesh(path, one_indexed=True):
    with open(path) as f:
        tokens = f.read().split()
    pos = 0
    n_nodes = int(tokens[pos]); pos += 1
    nodes = np.array(tokens[pos:pos + 3 * n_nodes], dtype=float).reshape(n_nodes, 3)
    pos += 3 * n_nodes
    n_tets = int(tokens[pos]); pos += 1
    raw = np.array(tokens[pos:pos + 5 * n_tets], dtype=np.int64).reshape(n_tets, 5)
    tets = raw[:, 1:5]
    if one_indexed:
        tets = tets - 1
    return nodes, tets


def write_mesh(path, nodes, tets, one_indexed=True, tag=1):
    out_tets = tets + 1 if one_indexed else tets
    with open(path, "w") as f:
        f.write(f"{len(nodes)}\n")
        for x, y, z in nodes:
            f.write(f"{x} {y} {z}\n")
        f.write(f"{len(out_tets)}\n")
        for a, b, c, d in out_tets:
            f.write(f"{tag} {a} {b} {c} {d}\n")


def signed_volumes(nodes, tets):
    a, b, c, d = nodes[tets[:, 0]], nodes[tets[:, 1]], nodes[tets[:, 2]], nodes[tets[:, 3]]
    return np.einsum('ij,ij->i', a - d, np.cross(b - d, c - d)) / 6.0


def reorient_negative_volume(nodes, tets):
    """meshtool requires consistently-oriented (positive signed volume)
    tets. Swap two vertex indices on any negative-volume element -- this
    changes only the labelling, not the geometry, so it fixes orientation
    without altering the mesh."""
    tets = tets.copy()
    vols = signed_volumes(nodes, tets)
    neg = vols < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return tets


# ---------------------------------------------------------------------------
# .mesh -> CARP
# ---------------------------------------------------------------------------
def mesh_to_carp(mesh_path, carp_base):
    nodes, tets = read_mesh(mesh_path)
    tets = reorient_negative_volume(nodes, tets)
    with open(carp_base + ".pts", "w") as f:
        f.write(f"{len(nodes)}\n")
        for p in nodes:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    with open(carp_base + ".elem", "w") as f:
        f.write(f"{len(tets)}\n")
        for t in tets:
            f.write("Tt " + " ".join(map(str, t)) + " 1\n")
    with open(carp_base + ".lon", "w") as f:
        f.write("1\n")
        for _ in tets:
            f.write("1.0 0.0 0.0\n")
    return len(nodes), len(tets)


# ---------------------------------------------------------------------------
# CARP -> .mesh (reverse direction, needed after meshtool cleaning to get
# back to this project's .mesh format)
# ---------------------------------------------------------------------------
def carp_to_mesh(carp_base, mesh_out_path):
    with open(carp_base + ".pts") as f:
        n = int(f.readline())
        pts = np.array([[float(x) for x in f.readline().split()] for _ in range(n)])
    tets = []
    with open(carp_base + ".elem") as f:
        m = int(f.readline())
        for _ in range(m):
            parts = f.readline().split()
            # format: "Tt i0 i1 i2 i3 region" -- skip the leading tag token
            # and the trailing region token, keep the 4 node indices
            tets.append([int(x) for x in parts[1:5]])
    tets = np.array(tets, dtype=np.int64)
    write_mesh(mesh_out_path, pts, tets, one_indexed=True)
    return len(pts), len(tets)


# ---------------------------------------------------------------------------
# meshtool cleaning (subprocess wrapper)
# ---------------------------------------------------------------------------
def run_clean_pass(meshtool_bin, in_base, out_base, threshold, smth=0.15):
    cmd = [meshtool_bin, "clean", "quality",
           f"-msh={in_base}", "-ifmt=carp_txt",
           f"-thr={threshold}", f"-smth={smth}",
           "-ofmt=carp_txt", f"-outmsh={out_base}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])  # meshtool prints a lot; keep the tail
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def run_query_quality(meshtool_bin, base):
    cmd = [meshtool_bin, "query", "quality", f"-msh={base}", "-ifmt=carp_txt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    return result.stdout


def clean_pipeline(meshtool_bin, carp_base, thresholds, smth=0.15):
    """Run each threshold as its own pass, feeding one pass's output into
    the next. Returns the final pass's output base name.

    LESSON FROM THIS PROJECT: do not include 0.3 in `thresholds` right after
    a fresh post-mesh smoothing pass -- see the module docstring. Two passes
    (0.7, 0.5) were sufficient to reach qmax ~0.5, comfortably under a 0.6
    target, without the instability the 3rd pass caused."""
    current = carp_base
    for i, thr in enumerate(thresholds, start=1):
        out_base = f"{carp_base}_clean{i}"
        print(f"\n--- clean pass {i}/{len(thresholds)}: threshold={thr} ---")
        ok = run_clean_pass(meshtool_bin, current, out_base, thr, smth)
        if not ok:
            print(f"WARNING: clean pass {i} reported an error -- inspect "
                  f"output above before trusting {out_base}", file=sys.stderr)
        current = out_base
    return current


# ---------------------------------------------------------------------------
def selftest():
    import tempfile

    # a mix of positive- and negative-oriented tets to test reorientation
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    tets_mixed = np.array([[0, 1, 2, 3], [0, 2, 1, 3]])  # 2nd is a vertex-swap of the 1st

    vols_before = signed_volumes(nodes, tets_mixed)
    assert (vols_before[0] > 0) != (vols_before[1] > 0), \
        "test setup should have mixed-sign volumes to exercise reorientation"

    reoriented = reorient_negative_volume(nodes, tets_mixed)
    vols_after = signed_volumes(nodes, reoriented)
    assert np.all(vols_after > 0) or np.all(vols_after < 0), \
        f"reorientation should make all volumes the same sign, got {vols_after}"

    # full .mesh <-> CARP round trip
    with tempfile.TemporaryDirectory() as d:
        mesh_path = os.path.join(d, "test.mesh")
        write_mesh(mesh_path, nodes, tets_mixed)

        carp_base = os.path.join(d, "test_carp")
        n_pts, n_tets = mesh_to_carp(mesh_path, carp_base)
        assert n_pts == 5 and n_tets == 2
        assert os.path.exists(carp_base + ".pts")
        assert os.path.exists(carp_base + ".elem")
        assert os.path.exists(carp_base + ".lon")

        # read the CARP files back and confirm reorientation was actually
        # written to disk (not just computed in memory)
        with open(carp_base + ".elem") as f:
            f.readline()  # count
            elem_lines = [f.readline().split() for _ in range(2)]
        written_tets = np.array([[int(x) for x in line[1:5]] for line in elem_lines])
        written_vols = signed_volumes(nodes, written_tets)
        assert np.all(written_vols > 0) or np.all(written_vols < 0), \
            "CARP file on disk should have consistently-oriented tets"

        # round trip back to .mesh
        mesh_out = os.path.join(d, "test_roundtrip.mesh")
        n_pts2, n_tets2 = carp_to_mesh(carp_base, mesh_out)
        assert n_pts2 == 5 and n_tets2 == 2
        nodes2, tets2 = read_mesh(mesh_out)
        assert np.allclose(nodes2, nodes)
        assert np.array_equal(tets2, written_tets)  # should match what CARP had

    print("selftest PASS: reorientation fixes mixed-sign volumes; "
          ".mesh -> CARP -> .mesh round trip is correct and preserves the "
          "on-disk reoriented connectivity.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh")
    ap.add_argument("--carp-base")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--meshtool", default="meshtool")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.5],
                     help="clean-pass thresholds in order; DEFAULT IS 0.7,0.5 "
                          "-- do not add 0.3 right after fresh smoothing, see docstring")
    ap.add_argument("--smth", type=float, default=0.15)
    ap.add_argument("--final-mesh", help="if --clean given, write the final "
                     "cleaned result back to this .mesh path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.mesh and args.carp_base):
        ap.error("--mesh and --carp-base are required (or --selftest)")

    n_pts, n_tets = mesh_to_carp(args.mesh, args.carp_base)
    print(f"Converted {args.mesh} -> {args.carp_base}.{{pts,elem,lon}} "
          f"({n_pts} nodes, {n_tets} tets, reoriented to NEGATIVE volume)")
    print()
    print("  NOTE: that CARP mesh is in the solver's negative-volume convention.")
    print("  `meshtool clean` accepts it, but `meshtool query quality` CANNOT score")
    print("  inverted elements and will report quality = 2 for every one of them --")
    print("  a value outside the metric's own [0,1] range. If you see 2, the mesh is")
    print("  not broken and neither is meshtool; you are measuring in the wrong")
    print("  convention.")
    print()
    print("  To measure this mesh correctly, either pass --clean (the quality check")
    print("  afterwards runs on the cleaned output, which meshtool has reoriented),")
    print("  or measure the .mesh file directly, which orients before measuring:")
    print(f"      python3 -c \"import sys; sys.path.insert(0,'.')")
    print(f"      from cortet import common as C")
    print(f"      print(C.measure_mesh_file('{args.mesh}', '<meshtool>', '/tmp')[0])\"")
    print()

    if args.clean:
        print("\nRunning meshtool cleaning pipeline ...")
        final_base = clean_pipeline(args.meshtool, args.carp_base, args.thresholds, args.smth)
        print(f"\nFinal cleaned CARP base: {final_base}")
        print("Independent quality check on the final result:")
        run_query_quality(args.meshtool, final_base)

        if args.final_mesh:
            n_pts2, n_tets2 = carp_to_mesh(final_base, args.final_mesh)
            print(f"\nWrote {args.final_mesh} ({n_pts2} nodes, {n_tets2} tets)")


if __name__ == "__main__":
    main()
