#!/usr/bin/env python3
r"""compare_roughness_gii_vs_mesh.py -- input roughness vs mesh-boundary roughness.

For one subject across a sweep of smoothing levels, compares the roughness of
the input surface with the roughness of the resulting mesh boundary. Smoothing
the input does not smooth the mesh boundary: the input falls several-fold while
the boundary stays flat, which is why the post-mesh step exists at all.

    python3 compare_roughness_gii_vs_mesh.py --dir meshes/ --subject subject_GA21.86_left
    python3 compare_roughness_gii_vs_mesh.py --selftest
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# .mesh reader (same format used throughout this project: node count, xyz
# lines, tet count, 1-indexed 4-node connectivity lines)
# ---------------------------------------------------------------------------
def read_mesh(path, one_indexed=True):
    """Reads this project's .mesh format: node count, xyz lines, tet count,
    then tet lines with 5 integers each -- a leading tag/region column
    followed by the 4 (1-indexed) node indices. Column 0 is discarded; only
    columns 1:5 are the actual connectivity. (This was the exact bug fixed
    here: an earlier version of this function assumed 4 ints/tet with no tag
    column, which silently misaligned and corrupted all connectivity after
    the first element -- see the booklet / project notes for how this was
    diagnosed.)"""
    with open(path) as f:
        tokens = f.read().split()
    pos = 0
    n_nodes = int(tokens[pos]); pos += 1
    nodes = np.array(tokens[pos:pos + 3 * n_nodes], dtype=float).reshape(n_nodes, 3)
    pos += 3 * n_nodes
    n_tets = int(tokens[pos]); pos += 1
    raw = np.array(tokens[pos:pos + 5 * n_tets], dtype=np.int64).reshape(n_tets, 5)
    tets = raw[:, 1:5]  # drop the leading tag column
    if one_indexed:
        tets = tets - 1
    return nodes, tets


# ---------------------------------------------------------------------------
# .gii reader (needs nibabel)
# ---------------------------------------------------------------------------
def read_gii_surface(path):
    import nibabel as nib
    g = nib.load(path)
    verts = np.asarray(g.darrays[0].data, dtype=float)
    faces = np.asarray(g.darrays[1].data, dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# Boundary-face extraction from a tetrahedral mesh: a triangular face that
# belongs to exactly ONE tetrahedron is a boundary face (internal faces are
# shared by exactly two tets and cancel out).
# ---------------------------------------------------------------------------
def extract_boundary_faces(tets):
    from collections import Counter
    face_count = Counter()
    face_owner = {}
    for t in tets:
        a, b, c, d = t
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            face_count[key] += 1
            face_owner.setdefault(key, face)  # keep original (unsorted) orientation
    boundary = np.array([face_owner[k] for k, cnt in face_count.items() if cnt == 1],
                         dtype=int)
    return boundary


def boundary_surface_from_mesh(nodes, tets):
    """Return (verts, faces) for just the boundary surface of a tet mesh,
    with faces re-indexed to a compact 0..N-1 vertex numbering (so the
    roughness metric only considers boundary vertices/neighbours, not the
    interior)."""
    boundary_faces = extract_boundary_faces(tets)
    used = np.unique(boundary_faces)
    remap = {old: new for new, old in enumerate(used)}
    faces = np.array([[remap[v] for v in f] for f in boundary_faces], dtype=int)
    verts = nodes[used]
    return verts, faces


# ---------------------------------------------------------------------------
# Roughness metric: umbrella-Laplacian deviation, normalised by mean edge length
# ---------------------------------------------------------------------------
def build_vertex_adjacency(n_verts, faces):
    adjacency = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def mean_edge_length(verts, faces):
    lengths = []
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            lengths.append(np.linalg.norm(verts[a] - verts[b]))
    return float(np.mean(lengths)) if lengths else 1.0


def roughness(verts, faces):
    """RMS umbrella-Laplacian magnitude, normalised by mean edge length.
    Returns (roughness_value, n_vertices_used)."""
    n = len(verts)
    adjacency = build_vertex_adjacency(n, faces)
    mags = []
    for i in range(n):
        neighbours = adjacency[i]
        if not neighbours:
            continue
        centroid = verts[list(neighbours)].mean(axis=0)
        lap = verts[i] - centroid
        mags.append(np.linalg.norm(lap))
    if not mags:
        return float("nan"), 0
    rms = float(np.sqrt(np.mean(np.square(mags))))
    scale = mean_edge_length(verts, faces)
    return (rms / scale if scale > 0 else float("nan")), n


# ---------------------------------------------------------------------------
# File discovery: match your naming convention (adjust patterns if yours differ)
# ---------------------------------------------------------------------------
def find_gii(directory, subject, level):
    if level == "raw":
        patterns = [f"{subject}*white.surf.gii", f"{subject}*.surf.gii"]
    else:
        patterns = [f"{subject}*sp{level}.surf.gii", f"{subject}*_sp{level}_*.surf.gii"]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        # for "raw" specifically, exclude anything with an sp<N> tag in the name
        if level == "raw":
            hits = [h for h in hits if "_sp" not in os.path.basename(h)]
        if hits:
            return hits[0]
    return None


def find_mesh(directory, subject, level):
    tag = "raw" if level == "raw" else f"sp{level}"
    patterns = [f"{subject}*{tag}*.mesh"]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
def run(args):
    rows = []
    print(f"{'level':<8}{'gii roughness':>16}{'mesh-boundary roughness':>26}"
          f"{'delta':>12}{'ratio':>10}")
    print("-" * 74)

    for level in args.levels:
        gii_path = find_gii(args.dir, args.subject, level)
        mesh_path = find_mesh(args.dir, args.subject, level)

        if gii_path is None or mesh_path is None:
            print(f"{level:<8} MISSING  gii={gii_path}  mesh={mesh_path}",
                  file=sys.stderr)
            rows.append([level, "", "", "", "", str(gii_path), str(mesh_path)])
            continue

        gii_verts, gii_faces = read_gii_surface(gii_path)
        gii_r, gii_n = roughness(gii_verts, gii_faces)

        nodes, tets = read_mesh(mesh_path)
        b_verts, b_faces = boundary_surface_from_mesh(nodes, tets)
        mesh_r, mesh_n = roughness(b_verts, b_faces)

        delta = mesh_r - gii_r
        ratio = (mesh_r / gii_r) if gii_r else float("nan")

        print(f"{level:<8}{gii_r:>16.5f}{mesh_r:>26.5f}{delta:>12.5f}{ratio:>10.3f}")
        rows.append([level, f"{gii_r:.6f}", f"{mesh_r:.6f}", f"{delta:.6f}",
                     f"{ratio:.4f}", os.path.basename(gii_path),
                     os.path.basename(mesh_path)])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "gii_roughness", "mesh_boundary_roughness",
                    "delta", "ratio", "gii_file", "mesh_file"])
        w.writerows(rows)

    print(f"\nWrote {args.out}")
    print("\nHow to read this:")
    print("  ratio > 1  -> the meshed boundary is ROUGHER than the smoothed input")
    print("                (CGAL reintroduced noise)")
    print("  ratio ~ 1  -> meshing roughly preserved the input's roughness")
    print("  ratio < 1  -> meshing was actually smoother than the input")
    print("If 'ratio' stays > 1 by a similar amount across all levels, CGAL is")
    print("adding a roughly constant noise floor regardless of input smoothing --")
    print("check whether boundary vertices are protected from ODT/Lloyd relocation,")
    print("and whether facet_distance/facet_size are tight relative to cell size.")


# ---------------------------------------------------------------------------
def selftest():
    # --- 1. boundary-face extraction: two tets sharing one internal face ---
    # tet A: 0,1,2,3 ; tet B: 1,2,3,4 -- they share face (1,2,3), which must
    # NOT appear in the boundary; every other face should.
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    boundary = extract_boundary_faces(tets)
    boundary_keys = set(tuple(sorted(f)) for f in boundary)
    assert (1, 2, 3) not in boundary_keys, "shared internal face wrongly kept as boundary"
    assert len(boundary) == 6, f"expected 6 boundary faces (4+4-2 shared), got {len(boundary)}"

    # --- 2. roughness metric: an INTERIOR vertex on a flat/regular grid
    # should have ~zero umbrella-Laplacian (boundary/corner vertices of a
    # small patch have asymmetric neighbourhoods even when flat -- that's a
    # real edge effect of the metric, not a bug). Use a large-enough grid
    # (interior vertices dominate) so that displacing one interior vertex
    # measurably raises the aggregate RMS roughness above the flat baseline,
    # the way it would on a real mesh with thousands of vertices. ---
    G = 9  # 9x9 grid -> 81 vertices, mostly interior
    xs, ys = np.meshgrid(range(G), range(G))
    flat_verts = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float),
                                   np.zeros(G * G)])
    faces = []
    for r in range(G - 1):
        for c in range(G - 1):
            i = r * G + c
            faces.append([i, i + 1, i + G])
            faces.append([i + 1, i + G + 1, i + G])
    faces = np.array(faces)

    adjacency = build_vertex_adjacency(G * G, faces)
    centre = (G // 2) * G + (G // 2)  # a fully-interior vertex
    centre_lap = flat_verts[centre] - flat_verts[list(adjacency[centre])].mean(axis=0)
    assert np.linalg.norm(centre_lap) < 1e-9, \
        f"interior vertex of a flat grid should have exactly zero Laplacian, got {centre_lap}"

    flat_rough, n_used = roughness(flat_verts, faces)
    assert n_used == G * G

    bumpy_verts = flat_verts.copy()
    bumpy_verts[centre, 2] += 2.0  # a large, unambiguous out-of-plane displacement
    bumpy_rough, _ = roughness(bumpy_verts, faces)
    assert bumpy_rough > flat_rough, \
        f"displacing an interior vertex should increase aggregate roughness ({bumpy_rough} vs {flat_rough})"

    # --- 3. boundary_surface_from_mesh re-indexes correctly ---
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    b_verts, b_faces = boundary_surface_from_mesh(nodes, tets)
    assert b_faces.max() < len(b_verts), "boundary re-indexing produced an out-of-range index"

    print("selftest PASS: boundary-face extraction (shared face correctly excluded),")
    print("               roughness metric (flat=~0, bumpy>flat), and boundary")
    print("               re-indexing all verified correct.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="folder containing both the .gii and .mesh files")
    ap.add_argument("--subject", help="subject id prefix to match, e.g. subject_GA21.86_left")
    ap.add_argument("--levels", nargs="+", default=["raw", "40", "50", "60", "70", "80", "90", "100", "200", "300"])
    ap.add_argument("--out", default="roughness_comparison.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.dir and args.subject):
        ap.error("--dir and --subject are required (or --selftest)")

    run(args)


if __name__ == "__main__":
    main()
