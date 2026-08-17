#!/usr/bin/env python3
r"""map_surface_fields_to_mesh.py -- carry a surface field onto the volume mesh.

Delaunay refinement replaces the input surface's vertices with a new node set,
so a per-vertex field (thickness, curvature, growth rate, labels) cannot be
carried across by index: doing so assigns the wrong value to every node, with
no error. This transfers it geometrically instead, by barycentric interpolation
onto the mesh boundary and nearest-neighbour extension into the interior.

    python3 map_surface_fields_to_mesh.py \
        --gii-surface surface.ico6.surf.gii \
        --mesh mesh.mesh --thickness thickness.shape.gii --out-dir fields/
"""

import numpy as np
import os
import sys
import time
import argparse

try:
    import nibabel as nib
except ImportError:
    print("ERROR: pip install -r requirements.txt  (needs nibabel)")
    sys.exit(1)

try:
    import trimesh
except ImportError:
    print("ERROR: pip install -r requirements.txt  (needs trimesh)")
    sys.exit(1)

from scipy.spatial import cKDTree


# ============================================================================
# Mesh I/O
# ============================================================================

def load_gii_surface(path):
    """Load GIfTI surface file → vertices (N,3), triangles (M,3)."""
    gii = nib.load(path)
    vertices = gii.darrays[0].data.astype(np.float64)
    triangles = gii.darrays[1].data.astype(np.int64)
    return vertices, triangles


def load_gii_scalar(path):
    """Load GIfTI scalar file → 1D array (N,)."""
    gii = nib.load(path)
    return gii.darrays[0].data.astype(np.float64)


def load_braingrowth_mesh(path):
    """
    Load BrainGrowth .mesh file → points (N,3), tets (M,4), faces (K,3).

    The .mesh format has surface nodes first (by construction from
    generate_tetrahedral_mesh.py), then interior nodes.

    Returns
    -------
    points : (n_nodes, 3)
    tets : (n_tets, 4) 0-based
    faces : (n_faces, 3) 0-based
    n_surface : int — number of surface nodes (first n_surface in points)
    """
    with open(path) as f:
        lines = f.readlines()

    idx = 0
    n_nodes = int(lines[idx].strip())
    idx += 1

    points = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        parts = lines[idx].strip().split()
        points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        idx += 1

    n_tets = int(lines[idx].strip())
    idx += 1

    tets = np.zeros((n_tets, 4), dtype=np.int64)
    for i in range(n_tets):
        parts = lines[idx].strip().split()
        # BrainGrowth format: tag n1 n2 n3 n4 (1-based)
        # Note: the file has n3/n4 pre-swapped for BrainGrowth loader convention
        # We read them as-is — we only need node positions, not tet orientation
        tets[i] = [int(parts[1]) - 1, int(parts[2]) - 1,
                    int(parts[3]) - 1, int(parts[4]) - 1]
        idx += 1

    n_faces = int(lines[idx].strip())
    idx += 1

    faces = np.zeros((n_faces, 3), dtype=np.int64)
    for i in range(n_faces):
        parts = lines[idx].strip().split()
        faces[i] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
        idx += 1

    # Surface nodes = unique nodes appearing in boundary faces
    surface_node_set = set(faces.flatten())
    n_surface = len(surface_node_set)

    # Verify that surface nodes are the first n_surface nodes
    # (guaranteed by generate_tetrahedral_mesh.py's reordering)
    max_surface_idx = max(surface_node_set)
    if max_surface_idx >= n_surface:
        print(f"  WARNING: Surface nodes not contiguous at start.")
        print(f"           max surface idx = {max_surface_idx}, n_surface = {n_surface}")
        print(f"           Mapping will still work but may be slower.")

    return points, tets, faces, n_surface


# ============================================================================
# Interpolator: project mesh surface nodes onto GIfTI surface
# ============================================================================

def build_interpolator(gii_vertices, gii_triangles, mesh_surface_points):
    """
    Project mesh surface nodes onto the GIfTI surface and compute
    barycentric coordinates for interpolation.

    For each mesh surface node:
      1. Find the nearest point on the GIfTI surface (triangle + position)
      2. Compute barycentric coordinates within that triangle

    Returns
    -------
    tri_indices : (n_surface,) — which GIfTI triangle each node projects onto
    bary_coords : (n_surface, 3) — barycentric weights [u, v, w]
    nearest_gii_vertex : (n_surface,) — index of closest GIfTI vertex
    distances : (n_surface,) — projection distance (mm)
    """
    print("  Building trimesh for GIfTI surface...")
    gii_mesh = trimesh.Trimesh(vertices=gii_vertices, faces=gii_triangles)

    print(f"  Projecting {len(mesh_surface_points)} mesh surface nodes...")
    t0 = time.time()

    # trimesh nearest-on-surface: returns closest point, distance, triangle index
    closest_points, distances, tri_indices = gii_mesh.nearest.on_surface(
        mesh_surface_points
    )

    elapsed = time.time() - t0
    print(f"  Projection: {elapsed:.1f}s")
    print(f"  Distance mesh→GIfTI: mean={distances.mean():.4f} mm, "
          f"max={distances.max():.4f} mm")

    if distances.max() > 1.0:
        print(f"  *** WARNING: max distance > 1mm — surfaces may not align! ***")

    # Compute barycentric coordinates (vectorised)
    print("  Computing barycentric coordinates...")
    n_surface = len(mesh_surface_points)
    bary_coords = np.zeros((n_surface, 3), dtype=np.float64)

    v0_idx = gii_triangles[tri_indices, 0]
    v1_idx = gii_triangles[tri_indices, 1]
    v2_idx = gii_triangles[tri_indices, 2]

    a = gii_vertices[v0_idx]  # (n_surface, 3)
    b = gii_vertices[v1_idx]
    c = gii_vertices[v2_idx]
    p = closest_points

    # Vectorised barycentric computation
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    # Guard against degenerate triangles
    denom_safe = np.where(np.abs(denom) < 1e-30, 1.0, denom)

    v_bary = (d11 * d20 - d01 * d21) / denom_safe
    w_bary = (d00 * d21 - d01 * d20) / denom_safe
    u_bary = 1.0 - v_bary - w_bary

    # Clamp and normalise
    u_bary = np.clip(u_bary, 0.0, 1.0)
    v_bary = np.clip(v_bary, 0.0, 1.0)
    w_bary = np.clip(w_bary, 0.0, 1.0)
    total = u_bary + v_bary + w_bary
    total = np.where(total > 0, total, 1.0)
    bary_coords[:, 0] = u_bary / total
    bary_coords[:, 1] = v_bary / total
    bary_coords[:, 2] = w_bary / total

    # Handle degenerate triangles (equal weights)
    degenerate = np.abs(denom) < 1e-30
    n_degen = np.sum(degenerate)
    if n_degen > 0:
        print(f"  {n_degen} degenerate triangles → equal-weight fallback")
        bary_coords[degenerate] = 1.0 / 3.0

    # Nearest GIfTI vertex (for discrete fields)
    # Pick the vertex with the largest barycentric weight
    max_bary_idx = np.argmax(bary_coords, axis=1)  # 0, 1, or 2
    tri_verts = gii_triangles[tri_indices]  # (n_surface, 3)
    nearest_gii_vertex = tri_verts[np.arange(n_surface), max_bary_idx]

    return tri_indices, bary_coords, nearest_gii_vertex, distances


def save_interpolator(path, tri_indices, bary_coords, nearest_gii_vertex, distances):
    """Save interpolator data for reuse."""
    np.savez_compressed(path,
                        tri_indices=tri_indices,
                        bary_coords=bary_coords,
                        nearest_gii_vertex=nearest_gii_vertex,
                        distances=distances)
    print(f"  Interpolator saved: {path}")


def load_interpolator(path):
    """Load precomputed interpolator."""
    data = np.load(path)
    print(f"  Interpolator loaded: {path}")
    return (data['tri_indices'],
            data['bary_coords'],
            data['nearest_gii_vertex'],
            data['distances'])


# ============================================================================
# Mapping functions
# ============================================================================

def map_continuous(gii_data, gii_triangles, tri_indices, bary_coords,
                   n_surface, n_total, mesh_points):
    """
    Map a continuous scalar field from GIfTI vertices to all mesh nodes.

    Surface nodes: barycentric interpolation from the GIfTI triangle they
    project onto.
    Interior nodes: nearest surface node value (depth within cortex is
    handled separately by dist_2_surf in the simulation).

    Parameters
    ----------
    gii_data : (n_gii_verts,) — scalar values at GIfTI vertices
    gii_triangles : (n_gii_tris, 3) — GIfTI triangle connectivity
    tri_indices : (n_surface,) — which GIfTI triangle each mesh surface node maps to
    bary_coords : (n_surface, 3) — barycentric weights
    n_surface : int — number of mesh surface nodes
    n_total : int — total mesh nodes
    mesh_points : (n_total, 3) — mesh node coordinates

    Returns
    -------
    all_values : (n_total,) — mapped scalar field
    """
    # Surface nodes: barycentric interpolation
    tri_verts = gii_triangles[tri_indices]  # (n_surface, 3)
    val_at_verts = gii_data[tri_verts]      # (n_surface, 3)
    surface_values = np.sum(val_at_verts * bary_coords, axis=1)

    # All nodes
    all_values = np.zeros(n_total, dtype=np.float64)
    all_values[:n_surface] = surface_values

    # Interior nodes: nearest surface node
    if n_total > n_surface:
        surface_coords = mesh_points[:n_surface]
        interior_coords = mesh_points[n_surface:]
        tree = cKDTree(surface_coords)
        _, nearest_idx = tree.query(interior_coords)
        all_values[n_surface:] = surface_values[nearest_idx]

    return all_values


def map_discrete(gii_data, nearest_gii_vertex, n_surface, n_total, mesh_points):
    """
    Map a discrete label field from GIfTI vertices to all mesh nodes.

    Surface nodes: nearest GIfTI vertex (no interpolation — labels are
    categorical, not continuous).
    Interior nodes: nearest surface node value.

    Parameters
    ----------
    gii_data : (n_gii_verts,) — label values at GIfTI vertices
    nearest_gii_vertex : (n_surface,) — closest GIfTI vertex per mesh surface node
    n_surface : int
    n_total : int
    mesh_points : (n_total, 3)

    Returns
    -------
    all_values : (n_total,) int32 — mapped labels
    """
    # Surface nodes: nearest GIfTI vertex
    surface_values = gii_data[nearest_gii_vertex].astype(np.int32)

    all_values = np.zeros(n_total, dtype=np.int32)
    all_values[:n_surface] = surface_values

    # Interior nodes: nearest surface node
    if n_total > n_surface:
        surface_coords = mesh_points[:n_surface]
        interior_coords = mesh_points[n_surface:]
        tree = cKDTree(surface_coords)
        _, nearest_idx = tree.query(interior_coords)
        all_values[n_surface:] = surface_values[nearest_idx]

    return all_values


def build_growth_from_labels(labels):
    """
    Build initial growth factor array from three-tier labels.

    Label 0 (medial wall) → gr = 0.0  (no growth)
    Label 1 (healthy)     → gr = 1.0  (full growth)
    Label 2 (noisy/filled)→ gr = 1.0  (full growth — these are real cortex)

    Later, when per-region weekly fits arrive, this function will be
    replaced by a region-aware lookup:
        gr[i] = region_params[regions[i]].growth_rate(week)

    Parameters
    ----------
    labels : (n_nodes,) int32 — three-tier labels

    Returns
    -------
    growth : (n_nodes,) float64 — growth factor per node
    """
    growth = np.ones(len(labels), dtype=np.float64)
    growth[labels == 0] = 0.0
    return growth


# ============================================================================
# Diagnostics
# ============================================================================

def print_mapping_summary(name, values, labels=None):
    """Print summary statistics for a mapped field."""
    print(f"\n  {name}:")
    print(f"    Shape: {values.shape}, dtype: {values.dtype}")
    print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
    print(f"    Mean:  {values.mean():.6f}, Std: {values.std():.6f}")

    if labels is not None:
        for lbl, lbl_name in [(0, "medial wall"), (1, "healthy"), (2, "noisy")]:
            mask = (labels == lbl)
            n = np.sum(mask)
            if n > 0:
                v = values[mask]
                print(f"    Label {lbl} ({lbl_name}): n={n}, "
                      f"mean={v.mean():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")


def print_region_summary(regions):
    """Print parcellation region distribution."""
    unique, counts = np.unique(regions, return_counts=True)
    print(f"\n  Parcellation regions:")
    print(f"    {len(unique)} unique regions")
    for r, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        pct = c / len(regions) * 100
        name = "medial wall" if r == 0 else ("unassigned" if r == -1 else f"region {r}")
        print(f"    Region {r:>3d} ({name}): {c:>6d} nodes ({pct:.1f}%)")
    if len(unique) > 10:
        print(f"    ... and {len(unique) - 10} more regions")


# ============================================================================
# Main pipeline
# ============================================================================

def run_mapping(args):
    """Run the full GIfTI → mesh mapping pipeline."""

    print("=" * 70)
    print("GIfTI → BrainGrowth Mesh Mapping Pipeline (v5)")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load GIfTI surface
    # ------------------------------------------------------------------
    print(f"\n[1] Loading GIfTI surface: {args.gii_surface}")
    gii_vertices, gii_triangles = load_gii_surface(args.gii_surface)
    n_gii = len(gii_vertices)
    print(f"    Vertices: {n_gii}, Triangles: {len(gii_triangles)}")

    # ------------------------------------------------------------------
    # Step 2: Load BrainGrowth mesh
    # ------------------------------------------------------------------
    print(f"\n[2] Loading BrainGrowth mesh: {args.mesh}")
    mesh_points, tets, faces, n_surface = load_braingrowth_mesh(args.mesh)
    n_total = len(mesh_points)
    n_interior = n_total - n_surface
    print(f"    Nodes: {n_total} ({n_surface} surface + {n_interior} interior)")
    print(f"    Tetrahedra: {len(tets)}, Boundary faces: {len(faces)}")

    # ------------------------------------------------------------------
    # Step 3: Build or load interpolator
    # ------------------------------------------------------------------
    interp_path = os.path.join(args.output_dir, "interpolator.npz")

    if args.load_interpolator and os.path.exists(args.load_interpolator):
        print(f"\n[3] Loading existing interpolator: {args.load_interpolator}")
        tri_indices, bary_coords, nearest_gii_vertex, distances = \
            load_interpolator(args.load_interpolator)

        # Sanity check
        if len(tri_indices) != n_surface:
            print(f"  ERROR: Interpolator has {len(tri_indices)} entries "
                  f"but mesh has {n_surface} surface nodes.")
            print(f"  Rebuild with --load-interpolator removed.")
            sys.exit(1)
    else:
        print(f"\n[3] Building interpolator (projecting mesh → GIfTI)...")
        mesh_surface_points = mesh_points[:n_surface]
        tri_indices, bary_coords, nearest_gii_vertex, distances = \
            build_interpolator(gii_vertices, gii_triangles, mesh_surface_points)
        save_interpolator(interp_path, tri_indices, bary_coords,
                          nearest_gii_vertex, distances)

    # ------------------------------------------------------------------
    # Step 4: Map scalar fields
    # ------------------------------------------------------------------
    print(f"\n[4] Mapping scalar fields to mesh nodes...")

    # --- 4a: Thickness (continuous, barycentric) ---
    if args.thickness:
        print(f"\n  Loading thickness: {args.thickness}")
        gii_thickness = load_gii_scalar(args.thickness)
        assert len(gii_thickness) == n_gii, \
            f"Thickness has {len(gii_thickness)} values, expected {n_gii}"

        thickness = map_continuous(gii_thickness, gii_triangles, tri_indices,
                                   bary_coords, n_surface, n_total, mesh_points)
        out_path = os.path.join(args.output_dir, "thickness.npy")
        np.save(out_path, thickness)
        print(f"  Saved: {out_path}")
    else:
        thickness = None
        print("  Thickness: skipped (no --thickness provided)")

    # --- 4b: Vertex labels (discrete, nearest-neighbour) ---
    if args.labels:
        print(f"\n  Loading vertex labels: {args.labels}")
        gii_labels = load_gii_scalar(args.labels)
        assert len(gii_labels) == n_gii

        labels = map_discrete(gii_labels, nearest_gii_vertex,
                              n_surface, n_total, mesh_points)
        out_path = os.path.join(args.output_dir, "labels.npy")
        np.save(out_path, labels)
        print(f"  Saved: {out_path}")
    else:
        labels = None
        print("  Vertex labels: skipped (no --labels provided)")

    # --- 4c: Parcellation regions (discrete, nearest-neighbour) ---
    if args.parcellation:
        print(f"\n  Loading parcellation: {args.parcellation}")
        gii_regions = load_gii_scalar(args.parcellation)
        assert len(gii_regions) == n_gii

        regions = map_discrete(gii_regions, nearest_gii_vertex,
                               n_surface, n_total, mesh_points)
        out_path = os.path.join(args.output_dir, "regions.npy")
        np.save(out_path, regions)
        print(f"  Saved: {out_path}")
    else:
        regions = None
        print("  Parcellation: skipped (no --parcellation provided)")

    # --- 4d: Growth factor (derived from labels for now) ---
    if labels is not None:
        print(f"\n  Building growth factor from labels...")
        growth = build_growth_from_labels(labels)
        out_path = os.path.join(args.output_dir, "growth.npy")
        np.save(out_path, growth)
        print(f"  Saved: {out_path}")
        print(f"  (Label 0 → gr=0, Labels 1,2 → gr=1)")
        print(f"  NOTE: When per-region fits arrive, growth will be "
              f"computed from region_params[region].growth_rate(week)")
    else:
        growth = None

    # ------------------------------------------------------------------
    # Step 5: Summary and diagnostics
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MAPPING SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  GIfTI vertices:     {n_gii}")
    print(f"  Mesh surface nodes: {n_surface}")
    print(f"  Mesh total nodes:   {n_total}")
    print(f"  Mesh tetrahedra:    {len(tets)}")
    print(f"  Projection accuracy: mean={distances.mean():.4f} mm, "
          f"max={distances.max():.4f} mm")

    if thickness is not None:
        print_mapping_summary("Thickness (mm)", thickness, labels)

    if growth is not None:
        print_mapping_summary("Growth factor", growth, labels)

    if labels is not None:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print(f"\n  Three-tier labels:")
        for lbl, cnt in zip(unique_labels, label_counts):
            name = {0: "medial wall", 1: "healthy", 2: "noisy"}.get(lbl, "?")
            pct = cnt / n_total * 100
            print(f"    Label {lbl} ({name}): {cnt} nodes ({pct:.1f}%)")

    if regions is not None:
        print_region_summary(regions)

    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Files:")
    for fname in ["interpolator.npz", "thickness.npy", "growth.npy",
                   "labels.npy", "regions.npy"]:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {fname:25s} ({size_kb:.0f} KB)")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Usage hint
    # ------------------------------------------------------------------
    print("  To use in a BrainGrowth-family solver (not included in this")
    print("  release; see the README's Scope section). For our own solver the")
    print("  invocation is:")
    print(f"    python3 BrainGrowth_v5.py \\")
    print(f"      -i {args.mesh} \\")
    print(f"      --per-vertex-dir {args.output_dir} \\")
    print(f"      -g 1.829 -sc 0.002 \\")
    print(f"      -o results/")
    print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map per-vertex data from GIfTI surface to BrainGrowth mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full mapping with all fields:
  python3 %(prog)s \\
      --gii-surface week28_left_white_asymmetry.ico6.surf.gii \\
      --mesh week28_brain.mesh \\
      --thickness week28_left_thickness_asymmetry.ico6.s3.shape.gii \\
      --labels week28_left_vertex_labels.ico6.shape.gii \\
      --parcellation week28_left_parcellation_manual.ico6.shape.gii \\
      --output-dir week28_mapped/

  # Reuse interpolator (fast — skips projection):
  python3 %(prog)s \\
      --gii-surface week28_left_white_asymmetry.ico6.surf.gii \\
      --mesh week28_brain.mesh \\
      --thickness week28_left_thickness_asymmetry.ico6.s3.shape.gii \\
      --output-dir week28_mapped/ \\
      --load-interpolator week28_mapped/interpolator.npz

  # Thickness only (minimal):
  python3 %(prog)s \\
      --gii-surface week28_left_white_asymmetry.ico6.surf.gii \\
      --mesh week28_brain.mesh \\
      --thickness week28_left_thickness_asymmetry.ico6.s3.shape.gii \\
      --output-dir week28_mapped/
        """,
    )
    parser.add_argument("--gii-surface", required=True,
                        help="GIfTI white surface (.surf.gii) — same one used for mesh generation")
    parser.add_argument("--mesh", required=True,
                        help="BrainGrowth .mesh file (output of generate_tetrahedral_mesh.py)")
    parser.add_argument("--thickness", default=None,
                        help="Three-tier processed thickness (.shape.gii)")
    parser.add_argument("--labels", default=None,
                        help="Three-tier vertex labels 0/1/2 (.shape.gii)")
    parser.add_argument("--parcellation", default=None,
                        help="Parcellation region IDs (.shape.gii)")
    parser.add_argument("--growth-data", default=None,
                        help="(Future) Per-vertex growth rate (.shape.gii). "
                             "If not provided, growth is derived from labels.")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .npy files")
    parser.add_argument("--load-interpolator", default=None,
                        help="Path to existing interpolator.npz to skip projection")
    args = parser.parse_args()

    # Validate
    for path_arg, name in [(args.gii_surface, "GIfTI surface"),
                           (args.mesh, "mesh")]:
        if not os.path.exists(path_arg):
            print(f"ERROR: {name} not found: {path_arg}", file=sys.stderr)
            sys.exit(1)

    for path_arg, name in [(args.thickness, "thickness"),
                           (args.labels, "labels"),
                           (args.parcellation, "parcellation"),
                           (args.growth_data, "growth-data")]:
        if path_arg is not None and not os.path.exists(path_arg):
            print(f"ERROR: {name} file not found: {path_arg}", file=sys.stderr)
            sys.exit(1)

    run_mapping(args)
