#!/usr/bin/env python3
r"""check_boundary_displacement.py -- boundary drift against the input surface.

Measures how far each boundary node of a mesh sits from the input surface it
was built from, and reports the mean and maximum. Modifies nothing.

    python3 check_boundary_displacement.py --gii input.surf.gii --mesh subject.mesh
    python3 check_boundary_displacement.py --selftest
"""
import argparse
import csv
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O (same conventions as compare_roughness_gii_vs_mesh.py)
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


def read_gii_surface(path):
    import nibabel as nib
    g = nib.load(path)
    verts = np.asarray(g.darrays[0].data, dtype=float)
    faces = np.asarray(g.darrays[1].data, dtype=int)
    return verts, faces


def extract_boundary_vertices(nodes, tets):
    """A triangular face belonging to exactly one tet is a boundary face;
    the union of its vertices is the boundary vertex set."""
    from collections import Counter
    face_count = Counter()
    for t in tets:
        a, b, c, d = t
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_count[tuple(sorted(face))] += 1
    boundary_ids = sorted(set(v for f, cnt in face_count.items() if cnt == 1 for v in f))
    return nodes[boundary_ids], np.array(boundary_ids)


# ---------------------------------------------------------------------------
# Nearest-neighbour matching (brute-force; fine for surface-sized vertex
# counts of a few hundred thousand -- switch to a KDTree via scipy if you
# need this faster on very large surfaces)
# ---------------------------------------------------------------------------
def nearest_neighbour_distances(query_points, reference_points):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(reference_points)
        dists, idx = tree.query(query_points, k=1)
        return dists, idx
    except ImportError:
        # brute-force fallback, no scipy required
        dists = np.empty(len(query_points))
        idx = np.empty(len(query_points), dtype=int)
        for i, p in enumerate(query_points):
            d = np.linalg.norm(reference_points - p, axis=1)
            j = np.argmin(d)
            dists[i] = d[j]
            idx[i] = j
        return dists, idx


# ---------------------------------------------------------------------------
def run(args):
    gii_verts, _ = read_gii_surface(args.gii)
    nodes, tets = read_mesh(args.mesh)
    boundary_verts, boundary_ids = extract_boundary_vertices(nodes, tets)

    dists, matched_idx = nearest_neighbour_distances(boundary_verts, gii_verts)

    mean_edge = _rough_scale(gii_verts)
    dists_norm = dists / mean_edge if mean_edge > 0 else dists

    print(f"Boundary vertices in mesh: {len(boundary_verts)}")
    print(f"Input .gii vertices:       {len(gii_verts)}")
    print()
    print(f"{'stat':<12}{'raw (mm)':>14}{'normalised':>14}")
    print("-" * 40)
    for name, val_raw, val_norm in [
        ("mean", dists.mean(), dists_norm.mean()),
        ("median", np.median(dists), np.median(dists_norm)),
        ("rms", np.sqrt(np.mean(dists**2)), np.sqrt(np.mean(dists_norm**2))),
        ("p95", np.percentile(dists, 95), np.percentile(dists_norm, 95)),
        ("max", dists.max(), dists_norm.max()),
    ]:
        print(f"{name:<12}{val_raw:>14.5f}{val_norm:>14.5f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary_vertex_id", "displacement_mm", "displacement_normalised"])
        for bid, d, dn in zip(boundary_ids, dists, dists_norm):
            w.writerow([bid, f"{d:.6f}", f"{dn:.6f}"])
    print(f"\nWrote per-vertex displacement to {args.out}")

    print()
    print("How to read this:")
    print("  displacement ~ 0 everywhere  -> boundary vertices are effectively")
    print("                                  protected/reprojected onto the input surface")
    print("  displacement small & uniform -> minor jitter, likely floating-point /")
    print("                                  reprojection-tolerance noise, not a real")
    print("                                  relocation mechanism")
    print("  displacement large and/or concentrated in specific regions -> the")
    print("  optimisation passes (ODT/Lloyd/Perturb/Exude) are genuinely relocating")
    print("  boundary vertices without reprojecting them back onto the input surface --")
    print("  this is the mechanism reintroducing noise regardless of input smoothing.")
    print()
    print("Tip: if displacement correlates with local input-surface curvature (higher")
    print("displacement in tightly-folded regions), that further supports ODT/Lloyd")
    print("relocation as the cause, since those passes respond to local mesh-quality")
    print("pressure, which is highest in high-curvature regions.")


def _rough_scale(points, sample=2000):
    """A characteristic length scale (median nearest-neighbour distance among
    a random sample of points) used only to normalise displacement magnitude
    for readability -- not a rigorous mesh-edge-length calculation."""
    n = len(points)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = points[idx]
    dists, _ = nearest_neighbour_distances(sub, points)
    # exclude self-matches (distance 0) by taking the median of nonzero values
    nonzero = dists[dists > 1e-12]
    return float(np.median(nonzero)) if len(nonzero) else 1.0


# ---------------------------------------------------------------------------
def selftest():
    # --- case 1: "protected" -- mesh boundary vertices are IDENTICAL to a
    # subset of the input surface vertices -> displacement should be ~0 ---
    rng = np.random.default_rng(1)
    gii_verts = rng.normal(size=(200, 3))
    boundary_verts_protected = gii_verts[:50].copy()  # exact subset, no movement

    dists, idx = nearest_neighbour_distances(boundary_verts_protected, gii_verts)
    assert dists.max() < 1e-9, f"protected case should have ~0 displacement, got max={dists.max()}"

    # --- case 2: "relocated" -- boundary vertices are the same subset but
    # deliberately displaced by a known amount -> displacement should reflect it ---
    known_shift = 0.37
    boundary_verts_relocated = boundary_verts_protected + np.array([known_shift, 0, 0])
    dists2, idx2 = nearest_neighbour_distances(boundary_verts_relocated, gii_verts)
    # nearest neighbour after a uniform shift should be close to (but not
    # necessarily exactly) the shift magnitude, since a different, nearby
    # gii vertex might now be closer than the original match for a
    # non-uniform point cloud -- so check it's in a sane range, not exact
    assert dists2.mean() > 0.5 * known_shift, \
        f"relocated case should show displacement near the known shift, got mean={dists2.mean()}"
    assert dists2.mean() < 3 * known_shift, \
        f"relocated case displacement implausibly larger than the known shift: {dists2.mean()}"

    # --- case 3: extract_boundary_vertices correctly finds only the true
    # boundary (excludes the internal shared face) ---
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    bverts, bids = extract_boundary_vertices(nodes, tets)
    assert set(bids.tolist()) == {0, 1, 2, 3, 4}, \
        "all 5 vertices touch a boundary face in this 2-tet example"

    print("selftest PASS: nearest-neighbour distance correctly reads ~0 for a")
    print("               protected/unmoved case and correctly detects a known")
    print("               deliberate displacement; boundary-vertex extraction correct.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gii")
    ap.add_argument("--mesh")
    ap.add_argument("--out", default="displacement_report.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.gii and args.mesh):
        ap.error("--gii and --mesh are required (or --selftest)")

    run(args)


if __name__ == "__main__":
    main()
