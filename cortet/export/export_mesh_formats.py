#!/usr/bin/env python3
r"""export_mesh_formats.py -- write a CORTET mesh out to other FE solver formats.

Stages 1-4 are solver-agnostic; only Stage 5 commits to one solver's
conventions. This writes the same volume to Abaqus, FEBio, FEniCS, VTK, Gmsh
or CARP via meshio (paper Section 3.3).

    python3 export_mesh_formats.py --mesh brain.mesh --out brain.inp
    python3 export_mesh_formats.py --mesh brain.mesh --out brain.msh --format gmsh22
    python3 export_mesh_formats.py --mesh brain.mesh --out brain.vtu --with-faces
    python3 export_mesh_formats.py --selftest
"""

import argparse
import os
import sys

import numpy as np

try:
    import meshio
except ImportError:
    print("ERROR: pip install -r requirements.txt  (needs meshio)")
    sys.exit(1)


# Suffix -> meshio format, for the targets the paper names. meshio can usually
# infer from the suffix; these are the cases where it cannot, or where more than
# one format shares a suffix.
SUFFIX_HINTS = {
    ".inp": "abaqus",
    ".feb": "febio",
    ".xdmf": "xdmf",      # the FEniCS route
    ".xmf": "xdmf",
    ".vtu": "vtu",
    ".vtk": "vtk",
    ".msh": "gmsh22",
}


def read_braingrowth_mesh(path):
    """
    Read the three-section .mesh format Stage 5 writes.

        <n_nodes>
        x y z                       (surface nodes first, then interior)
        <n_tets>
        tag n1 n2 n3 n4             (1-based; n3/n4 pre-swapped for the solver)
        <n_faces>
        tag f1 f2 f3                (1-based boundary triangles)

    Returns (points, tets, faces), all 0-based. `faces` is None if the file has
    no third section.

    Note the five fields per tet line: tag first, then four node indices. Reading
    four integers instead of five silently shifts every index by one column and
    yields a mesh that looks valid and is not.
    """
    with open(path) as fh:
        lines = fh.readlines()

    idx = 0
    n_nodes = int(lines[idx].split()[0]); idx += 1
    points = np.empty((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        parts = lines[idx].split(); idx += 1
        points[i] = (float(parts[0]), float(parts[1]), float(parts[2]))

    n_tets = int(lines[idx].split()[0]); idx += 1
    tets = np.empty((n_tets, 4), dtype=np.int64)
    for i in range(n_tets):
        parts = lines[idx].split(); idx += 1
        if len(parts) < 5:
            raise ValueError(
                f"{path}: tet line {idx} has {len(parts)} fields, expected 5 "
                f"(tag n1 n2 n3 n4). This is not the .mesh format this pipeline "
                f"writes.")
        tets[i] = [int(parts[1]) - 1, int(parts[2]) - 1,
                   int(parts[3]) - 1, int(parts[4]) - 1]

    faces = None
    if idx < len(lines) and lines[idx].strip():
        n_faces = int(lines[idx].split()[0]); idx += 1
        faces = np.empty((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            parts = lines[idx].split(); idx += 1
            faces[i] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]

    return points, tets, faces


def read_carp(basename):
    """Read a CARP text mesh (.pts/.elem). Returns (points, tets), 0-based."""
    with open(f"{basename}.pts") as fh:
        n_nodes = int(fh.readline().split()[0])
        points = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            points[i] = [float(x) for x in fh.readline().split()[:3]]

    with open(f"{basename}.elem") as fh:
        n_tets = int(fh.readline().split()[0])
        tets = np.empty((n_tets, 4), dtype=np.int64)
        for i in range(n_tets):
            parts = fh.readline().split()
            # Tt n0 n1 n2 n3 tag   (already 0-based in CARP)
            tets[i] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]

    return points, tets


def signed_volumes(points, tets):
    """Six times the signed volume of each tet. Negative means inverted."""
    p = points[tets]
    return np.einsum("ij,ij->i",
                     p[:, 1] - p[:, 0],
                     np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))


def export(points, tets, out_path, fmt=None, faces=None):
    cells = [("tetra", tets)]
    if faces is not None:
        cells.append(("triangle", faces))

    if fmt is None:
        fmt = SUFFIX_HINTS.get(os.path.splitext(out_path)[1].lower())

    mesh = meshio.Mesh(points=points, cells=cells)
    if fmt:
        meshio.write(out_path, mesh, file_format=fmt)
    else:
        meshio.write(out_path, mesh)   # let meshio infer
    return fmt


def selftest():
    """Round-trip a single tetrahedron through the .mesh reader and an export."""
    import tempfile

    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # write it the way Stage 5 does: 1-based, tag first, n3/n4 swapped
    with tempfile.TemporaryDirectory() as d:
        mesh_path = os.path.join(d, "one.mesh")
        with open(mesh_path, "w") as fh:
            fh.write("4\n")
            for p in pts:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            fh.write("1\n   1         1         2         4         3\n")
            fh.write("1\n   1         1         2         3\n")

        p2, t2, f2 = read_braingrowth_mesh(mesh_path)
        assert p2.shape == (4, 3), p2.shape
        assert t2.tolist() == [[0, 1, 3, 2]], t2.tolist()
        assert f2.tolist() == [[0, 1, 2]], f2.tolist()
        print("  read_braingrowth_mesh: 5-field tet line parsed correctly     OK")

        # the file is in solver convention, so as-read the volume is negative
        assert signed_volumes(p2, t2)[0] < 0
        t2[:, [2, 3]] = t2[:, [3, 2]]
        assert signed_volumes(p2, t2)[0] > 0
        print("  n3<->n4 unswap restores positive orientation                 OK")

        out = os.path.join(d, "one.vtu")
        export(p2, t2, out)
        back = meshio.read(out)
        assert len(back.points) == 4
        print("  export -> vtu -> meshio.read round-trip                      OK")

        # a truncated tet line must be rejected, not silently misread
        bad = os.path.join(d, "bad.mesh")
        with open(bad, "w") as fh:
            fh.write("4\n")
            for p in pts:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            fh.write("1\n   1 2 4 3\n")
        try:
            read_braingrowth_mesh(bad)
        except ValueError:
            print("  4-field tet line rejected rather than misparsed           OK")
        else:
            raise AssertionError("4-field tet line was accepted")

    print("\nselftest passed.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mesh", help="input BrainGrowth .mesh file")
    src.add_argument("--carp", help="input CARP basename (no extension)")
    ap.add_argument("--out", help="output path; format inferred from the suffix")
    ap.add_argument("--format", default=None,
                    help="meshio format name, if the suffix is ambiguous "
                         "(abaqus, febio, xdmf, vtu, vtk, gmsh22, ...)")
    ap.add_argument("--with-faces", action="store_true",
                    help="also write the boundary triangles as a second cell "
                         "block (VTK/VTU/Gmsh; some solver formats reject it)")
    ap.add_argument("--no-swap", action="store_true",
                    help="do NOT undo the n3<->n4 solver swap when reading a "
                         ".mesh file. Only correct if the input was written "
                         "in standard orientation already")
    ap.add_argument("--list-formats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if args.list_formats:
        # meshio does not expose a public list, and the private attribute has
        # moved between versions, so fall back to the formats the paper names.
        names = None
        for attr in ("_writer_map", "_helpers"):
            obj = getattr(meshio, attr, None)
            if obj is not None:
                names = getattr(obj, "writer_map", obj) if attr == "_helpers" else obj
                break
        if isinstance(names, dict):
            for name in sorted(names):
                print(" ", name)
        else:
            print("  meshio's format list is not exposed by this version. The "
                  "formats the paper names are:")
            for name in ("abaqus", "febio", "xdmf (FEniCS)", "vtk", "vtu",
                         "gmsh", "gmsh22"):
                print("   ", name)
        return
    if not (args.mesh or args.carp) or not args.out:
        ap.error("need --mesh or --carp, plus --out")

    faces = None
    if args.mesh:
        points, tets, faces = read_braingrowth_mesh(args.mesh)
        if not args.no_swap:
            tets = tets.copy()
            tets[:, [2, 3]] = tets[:, [3, 2]]
            print("  Undid the Stage-5 n3<->n4 swap (standard orientation).")
    else:
        points, tets = read_carp(args.carp)

    vols = signed_volumes(points, tets)
    n_neg = int((vols <= 0).sum())
    print(f"  Nodes {len(points):,}   tets {len(tets):,}"
          + (f"   boundary faces {len(faces):,}" if faces is not None else ""))
    print(f"  Tets with non-positive signed volume: {n_neg:,}"
          + ("" if n_neg == 0 else "   <-- CHECK THE INPUT CONVENTION"))

    fmt = export(points, tets, args.out, args.format,
                 faces if args.with_faces else None)
    print(f"  Wrote {args.out}" + (f"  (format: {fmt})" if fmt else ""))


if __name__ == "__main__":
    main()
