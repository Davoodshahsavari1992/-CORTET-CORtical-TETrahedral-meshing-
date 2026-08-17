#!/usr/bin/env python3
r"""generate_tetrahedral_mesh.py -- GIfTI surface in, solver-ready mesh out.

The six stages of the paper's Figure 1:

    S1  CGAL Delaunay refinement, all four optimisation passes (pygalmesh)
    S2  convert to CARP (meshio)
    S3  global vertex smoothing, 5 passes (Gmsh Relocate3D)            optional
    S4  worst-element cleaning, 3 passes at q > 0.7, 0.5, 0.3          optional
    S5  solver-ready export: boundary faces, node ordering, conventions
    S6  snap the boundary back onto the input surface                  optional

The single control is --cell-size h, the upper bound on each tetrahedron's
circumradius and on the surface Delaunay ball radius, with facet distance h/2.
Default 0.6 mm, the resolution the cohort results were produced at.

    python3 generate_tetrahedral_mesh.py -i input.surf.gii -o output.mesh
    python3 generate_tetrahedral_mesh.py -i input.surf.gii -o output.mesh --cell-size 0.4
    python3 generate_tetrahedral_mesh.py -i input.surf.gii -o output.mesh --skip-gmsh
    python3 generate_tetrahedral_mesh.py -i input.surf.gii -o output.mesh \
        --meshtool-path /path/to/meshtool
"""

import nibabel as nib
import numpy as np
import trimesh
import pygalmesh
import meshio
import tempfile
import subprocess
import os
import sys
import argparse
import time


# ============================================================================
# STAGE 1: Load GIfTI and generate tetrahedral volume mesh with pygalmesh
# ============================================================================

def stage1_generate_volume_mesh(input_gii, cell_size=0.6, work_dir="/tmp"):
    """
    Load GIfTI surface → export STL → pygalmesh tetrahedralization.

    All four CGAL optimisation passes are enabled: ODT and Lloyd relocate
    interior vertices to regularise element size and shape, perturb targets the
    worst-shaped elements, and exude removes slivers by reweighting vertices in
    place — which is why exude must run last.

    Historical note: under CGAL 5.0.3 built with gcc-13.2.0, odt/lloyd/perturb
    segfaulted and exude alone was usable. That is fixed from CGAL 5.6.1, which
    this pipeline requires. If you hit a segfault here, check your CGAL version
    before anything else.

    Parameters
    ----------
    input_gii : str
        Path to input .gii surface file.
    cell_size : float
        Controls mesh density, in mm. 0.6 is the paper's default resolution and
        gives ~0.3-2.1M tets across the fetal age range.
        Smaller = more tets (0.4 → roughly 3x, 0.8 → roughly 0.4x).
    work_dir : str
        Directory for temporary files.
    
    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Mesh node coordinates.
    tets : np.ndarray, shape (M, 4)
        Tetrahedral connectivity (0-based indexing).
    """
    print("=" * 70)
    print("STAGE 1: pygalmesh volume mesh generation with exude optimization")
    print("=" * 70)
    
    # Load GIfTI
    gii = nib.load(input_gii)
    vertices = gii.darrays[0].data   # (N, 3) float32
    faces = gii.darrays[1].data      # (M, 3) int32
    
    bbox_range = vertices.max(axis=0) - vertices.min(axis=0)
    print(f"  Surface: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"  Bounding box range: [{bbox_range[0]:.1f}, {bbox_range[1]:.1f}, {bbox_range[2]:.1f}] mm")
    print(f"  Cell size: {cell_size}")
    
    # Export as STL via trimesh (fixes normals for CGAL)
    surface = trimesh.Trimesh(vertices=vertices, faces=faces)
    surface.fix_normals()
    
    tmp_stl = os.path.join(work_dir, "brain_surface_tmp.stl")
    surface.export(tmp_stl)
    print(f"  Temporary STL: {tmp_stl}")
    
    # Generate tetrahedral volume mesh
    # Parameters scaled to brain mesh in mm coordinates (~30-80mm range)
    print("  Generating volume mesh (this may take a few minutes)...")
    t0 = time.time()
    
    mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
        tmp_stl,
        min_facet_angle=25.0,
        max_radius_surface_delaunay_ball=cell_size,
        max_facet_distance=cell_size / 2.0,
        max_circumradius_edge_ratio=2.0,
        max_cell_circumradius=cell_size,
        # CGAL 5.6.1 — all four optimizations work
        # (CGAL 5.0.3 had segfault bugs in lloyd/odt/perturb with gcc-13.2.0)
        odt=True,
        lloyd=True,
        perturb=True,
        exude=True,  # sliver exudation (must be last in chain)
        verbose=True,
    )
    
    elapsed = time.time() - t0
    print(f"  Generation time: {elapsed:.1f}s")
    
    # Clean up temp file
    os.remove(tmp_stl)
    
    # Extract tetrahedra (mesh may also contain surface triangles)
    points = mesh.points
    tets = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            break
    
    if tets is None:
        raise RuntimeError("No tetrahedra found in pygalmesh output!")
    
    print(f"  Result: {points.shape[0]} nodes, {tets.shape[0]} tets")
    print()
    
    return points, tets


# ============================================================================
# STAGE 2: Convert to CARP format for meshtool
# ============================================================================

def write_carp(points, tets, basename):
    """
    Write mesh in CARP text format (.pts, .elem, .lon).
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    tets : np.ndarray, shape (M, 4), 0-based indexing
    basename : str
        Output basename (without extension).
    """
    n_nodes = points.shape[0]
    n_tets = tets.shape[0]
    
    # .pts file
    with open(f"{basename}.pts", "w") as f:
        f.write(f"{n_nodes}\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    
    # .elem file
    with open(f"{basename}.elem", "w") as f:
        f.write(f"{n_tets}\n")
        for t in tets:
            f.write(f"Tt {t[0]} {t[1]} {t[2]} {t[3]} 1\n")
    
    # .lon file (fiber directions, required by meshtool)
    with open(f"{basename}.lon", "w") as f:
        f.write("1\n")
        for _ in range(n_tets):
            f.write("1.000000 0.000000 0.000000\n")


def read_carp(basename):
    """
    Read mesh from CARP text format (.pts, .elem).
    
    Returns
    -------
    points : np.ndarray, shape (N, 3)
    tets : np.ndarray, shape (M, 4), 0-based indexing
    """
    # Read .pts
    with open(f"{basename}.pts") as f:
        n_nodes = int(f.readline().strip())
        points = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            points[i] = [float(x) for x in f.readline().split()]
    
    # Read .elem
    with open(f"{basename}.elem") as f:
        n_tets = int(f.readline().strip())
        tets = np.zeros((n_tets, 4), dtype=int)
        for i in range(n_tets):
            parts = f.readline().split()
            # Format: Tt n0 n1 n2 n3 tag
            tets[i] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
    
    return points, tets


# ============================================================================
# STAGE 3: Gmsh Relocate3D global vertex smoothing
# ============================================================================

def stage3_gmsh_relocate(basename_in, basename_out, n_passes=5):
    """
    Run Gmsh Relocate3D optimization (global vertex relocation).

    On some Linux HPC systems gmsh needs Mesa/GLU on LD_LIBRARY_PATH; point
    CORTET_GL_LIBS at those directories if `import gmsh` fails. If Gmsh is not
    available at all, this stage is skipped gracefully.
    
    Parameters
    ----------
    basename_in : str
        Input CARP mesh basename.
    basename_out : str
        Output CARP mesh basename.
    n_passes : int
        Number of Relocate3D passes (default: 5).
    
    Returns
    -------
    success : bool
        True if Gmsh optimization ran, False if skipped.
    """
    print("=" * 70)
    print(f"STAGE 3: Gmsh Relocate3D optimization ({n_passes} passes)")
    print("=" * 70)
    
    # On some Linux HPC systems the gmsh wheel needs Mesa/GLU on LD_LIBRARY_PATH
    # and the module system does not put it there. Set CORTET_GL_LIBS to a
    # colon-separated list of those lib directories if `import gmsh` fails with
    # a missing libGLU/libGL. Unset on a normal desktop install, where the
    # system libraries are already found. See docs/ENVIRONMENT.md.
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
        print("  Skipping Gmsh optimization.")
        print()
        return False
    
    # Convert CARP → Gmsh .msh format via meshio
    print("  Converting CARP → Gmsh format...")
    points, tets = read_carp(basename_in)
    
    msh_in = f"{basename_in}.msh"
    msh_out = f"{basename_out}.msh"
    
    mesh_obj = meshio.Mesh(points=points, cells=[("tetra", tets)])
    meshio.write(msh_in, mesh_obj, file_format="gmsh22")
    
    # Run Relocate3D passes
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
        print("  Skipping Gmsh optimization.")
        try:
            gmsh.finalize()
        except:
            pass
        # Clean up temp files
        for f in [msh_in, msh_out]:
            if os.path.exists(f):
                os.remove(f)
        return False
    
    elapsed = time.time() - t0
    print(f"  Gmsh time: {elapsed:.1f}s")
    
    # Convert back Gmsh → CARP
    print("  Converting Gmsh → CARP format...")
    gmsh_mesh = meshio.read(msh_out)
    gmsh_pts = gmsh_mesh.points
    gmsh_tets = None
    for cell_block in gmsh_mesh.cells:
        if cell_block.type == "tetra":
            gmsh_tets = cell_block.data
            break
    
    if gmsh_tets is None:
        print("  ERROR: No tetrahedra in Gmsh output. Skipping.")
        return False
    
    write_carp(gmsh_pts, gmsh_tets, basename_out)
    print(f"  Result: {gmsh_pts.shape[0]} nodes, {gmsh_tets.shape[0]} tets")
    
    # Clean up .msh files
    for f in [msh_in, msh_out]:
        if os.path.exists(f):
            os.remove(f)
    
    print()
    return True


# ============================================================================
# STAGE 4: Iterative meshtool quality cleaning (3 passes)
# ============================================================================

def stage4_meshtool_clean(basename_in, basename_out, meshtool_path, thresholds=None):
    """
    Run iterative meshtool quality cleaning with decreasing thresholds.
    
    Meshtool quality metric: tet_qmetric_volume (0=best, 1=worst).
    Each pass shifts vertices to reduce elements above the threshold.
    
    Parameters
    ----------
    basename_in : str
        Input CARP mesh basename.
    basename_out : str
        Final output CARP mesh basename.
    meshtool_path : str
        Path to meshtool binary.
    thresholds : list of float
        Quality thresholds for each pass. Default: [0.7, 0.5, 0.3].
    """
    print("=" * 70)
    print("STAGE 4: Iterative meshtool quality cleaning")
    print("=" * 70)
    
    if thresholds is None:
        thresholds = [0.7, 0.5, 0.3]
    
    current_input = basename_in

    for i, thr in enumerate(thresholds):
        pass_num = i + 1
        if pass_num < len(thresholds):
            current_output = f"{basename_out}_pass{pass_num}"
        else:
            current_output = basename_out

        print(f"\n  Pass {pass_num}/{len(thresholds)}: threshold={thr}")

        cmd = [
            meshtool_path, "clean", "quality",
            f"-msh={current_input}",
            "-ifmt=carp_txt",
            f"-outmsh={current_output}",
            "-ofmt=carp_txt",
            f"-thr={thr}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output for quality stats
        for out_line in result.stdout.split("\n"):
            if "min:" in out_line.lower() and "max:" in out_line.lower():
                print(f"    {out_line.strip()}")
            if "Number of elements" in out_line:
                print(f"    {out_line.strip()}")

        # Check if output was actually written.
        #
        # BEWARE: "no output file" has TWO causes that look identical here --
        # meshtool ran and found nothing to fix, or meshtool FAILED. This branch
        # used to assume the first and copy the input forward, printing "mesh
        # already clean". On a run where meshtool was broken it therefore
        # reported success on all three passes while doing nothing at all, and
        # the pipeline carried an uncleaned mesh to the end without a single
        # warning. That is precisely the silent-failure class this project
        # exists to eliminate, so the two cases are now told apart.
        if not os.path.exists(f"{current_output}.pts"):
            import shutil

            # Evidence that meshtool actually ran and had nothing to do.
            ran_ok = result.returncode == 0
            said_clean = any(
                "elements above threshold: 0" in ln.replace("  ", " ")
                or "number of elements above threshold: 0" in ln.lower()
                for ln in result.stdout.split("\n"))
            produced_output = bool(result.stdout.strip())

            if ran_ok and (said_clean or produced_output):
                print(f"  No output written — meshtool reports nothing above "
                      f"this threshold; carrying the mesh forward unchanged.")
            else:
                print(f"\n  *** meshtool PRODUCED NO OUTPUT AND NO EVIDENCE OF "
                      f"RUNNING ***", file=sys.stderr)
                print(f"  exit code : {result.returncode}", file=sys.stderr)
                if result.stderr.strip():
                    print(f"  stderr    : {result.stderr.strip()[:400]}",
                          file=sys.stderr)
                print(f"  This pass did NOT clean the mesh. Continuing would "
                      f"produce an\n  uncleaned mesh reported as successful. "
                      f"Check the module environment\n  (docs/ENVIRONMENT.md) "
                      f"and run run/00_check_environment.py.\n", file=sys.stderr)
                raise RuntimeError(
                    f"meshtool clean pass {pass_num} (threshold {thr}) produced "
                    f"no output and no evidence of having run")

            for ext in [".pts", ".elem", ".lon"]:
                src = f"{current_input}{ext}"
                dst = f"{current_output}{ext}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        current_input = current_output

    print()
def query_meshtool_quality(basename, meshtool_path):
    """Run meshtool quality query and return (mean, max, stddev)."""
    cmd = [
        meshtool_path, "query", "quality",
        f"-msh={basename}",
        "-ifmt=carp_txt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    mean_val = max_val = stddev_val = None
    for line in result.stdout.split("\n"):
        if line.startswith("Min:"):
            # Format: Min: 0.000568358 Max: 0.997225 Mean: 0.244539 Stddev: 0.161467
            parts = line.split()
            for j, p in enumerate(parts):
                if p == "Mean:":
                    mean_val = float(parts[j + 1])
                elif p == "Max:":
                    max_val = float(parts[j + 1])
                elif p == "Stddev:":
                    stddev_val = float(parts[j + 1])
    
    return mean_val, max_val, stddev_val


# ============================================================================
# STAGE 5: Convert to BrainGrowth .mesh format (with tet swap + boundary faces)
# ============================================================================

def write_braingrowth_mesh(points, tets, output_path):
    """
    Write mesh in BrainGrowth-compatible format (Tallinen .mesh format).
    
    BrainGrowth's mesh loader has specific conventions:
    1. get_nodes() swaps x<->y when reading: coords[i] = [mesh[1], mesh[0], mesh[2]]
    2. get_tetra_indices() swaps n3<->n4: tets[i] = [n1-1, n2-1, n4-1, n3-1]
    3. volume_mesh() returns -sum(det/6), expecting negative determinants
    4. get_face_indices() reads a surface triangle section after the tets
    5. tetra_normals() indexes surf_node_norms with nearest_surf_node (global node IDs),
       which only works if surface nodes are the FIRST nodes in the mesh (indices 0..N_surf-1)
    
    To be compatible, we:
    - Reorder nodes: surface nodes first (0..N_surf-1), then interior nodes
    - Swap n3<->n4 in the tet output so BrainGrowth's swap produces correct orientation
    - Extract and append boundary faces (required for contact detection & surface normals)
    
    Format:
        <n_nodes>
        x1 y1 z1           (surface nodes first, then interior)
        ...
        <n_tets>
        tag  n1  n2  n4  n3    (1-based, n3/n4 pre-swapped for BrainGrowth)
        ...
        <n_faces>
        tag  f1  f2  f3        (1-based, boundary triangles)
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    tets : np.ndarray, shape (M, 4), 0-based indexing
    output_path : str
        Output .mesh file path.
    """
    from collections import defaultdict
    
    n_nodes = points.shape[0]
    n_tets = tets.shape[0]
    
    # --- Step 1: Extract boundary faces (faces shared by exactly 1 tet) ---
    face_count = defaultdict(list)
    for tet in tets:
        for combo in [(tet[0],tet[1],tet[2]), (tet[0],tet[1],tet[3]),
                      (tet[0],tet[2],tet[3]), (tet[1],tet[2],tet[3])]:
            face_count[tuple(sorted(combo))].append(combo)
    
    boundary_faces = []
    for key, combos in face_count.items():
        if len(combos) == 1:
            boundary_faces.append(combos[0])
    
    boundary_faces = np.array(boundary_faces)
    n_faces = len(boundary_faces)
    
    # --- Step 2: Identify surface nodes ---
    is_surface = np.zeros(n_nodes, dtype=bool)
    for face in boundary_faces:
        is_surface[face[0]] = is_surface[face[1]] = is_surface[face[2]] = True
    
    surface_nodes = np.where(is_surface)[0]
    interior_nodes = np.where(~is_surface)[0]
    n_surface = len(surface_nodes)
    n_interior = len(interior_nodes)
    
    # --- Step 3: Build reordering map (surface first, then interior) ---
    # This is required because BrainGrowth's tetra_normals() indexes
    # surf_node_norms with global node IDs from nearest_surf_node.
    # Tallinen's mesh has surface nodes as the first N_surf nodes.
    old_to_new = np.zeros(n_nodes, dtype=int)
    new_order = np.concatenate([surface_nodes, interior_nodes])
    for new_idx, old_idx in enumerate(new_order):
        old_to_new[old_idx] = new_idx
    
    # Reorder coordinates
    new_points = points[new_order]
    
    # Remap tet and face indices
    new_tets = old_to_new[tets]
    new_faces = old_to_new[boundary_faces]
    
    print(f"  Node reordering: {n_surface} surface nodes first, then {n_interior} interior")
    print(f"  Surface node index range: [0, {n_surface-1}]")
    print(f"  Boundary faces extracted: {n_faces}")
    
    # --- Step 4: Write mesh file ---
    with open(output_path, "w") as f:
        # Node section (reordered: surface first)
        f.write(f"{n_nodes}\n")
        for p in new_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        # Tet section — swap n3<->n4 for BrainGrowth compatibility
        # BrainGrowth reads [n1, n2, n4, n3] so we write [n1, n2, n4, n3]
        # so that after its swap it gets correct negative-det orientation
        f.write(f"{n_tets}\n")
        for t in new_tets:
            f.write(f"   1  {t[0]+1:>8}  {t[1]+1:>8}  {t[3]+1:>8}  {t[2]+1:>8}\n")
        
        # Face section — boundary triangles for contact detection
        f.write(f"{n_faces}\n")
        for face in new_faces:
            f.write(f"   1  {face[0]+1:>8}  {face[1]+1:>8}  {face[2]+1:>8}\n")


# ============================================================================
# STAGE 6 (optional): Reproject boundary vertices to original GIfTI surface
# ============================================================================

def stage6_reproject_boundary(input_gii, points, tets):
    """
    Reproject boundary vertices of the tetrahedral mesh back onto the
    original GIfTI surface. This eliminates boundary drift introduced by
    CGAL's surface remeshing during Delaunay refinement.
    
    CGAL's generate_volume_mesh_from_surface_mesh() creates new surface
    vertices that approximate but don't exactly match the input surface.
    This function projects those boundary vertices to the nearest point
    on the original surface.
    
    Typical drift before reprojection: ~0.33mm mean, ~0.75mm max.
    After reprojection: boundary lies exactly on original surface.
    
    Trade-off: slightly degrades element quality near the surface
    (mean ~0.129→0.135, max ~0.491→0.500), but still beats Tallinen reference.
    
    Parameters
    ----------
    input_gii : str
        Path to original GIfTI surface file.
    points : np.ndarray, shape (N, 3)
        Mesh node coordinates (modified in-place).
    tets : np.ndarray, shape (M, 4), 0-based indexing.
    
    Returns
    -------
    points : np.ndarray
        Reprojected node coordinates.
    n_inverted : int
        Number of inverted tets after reprojection (should be 0).
    """
    import nibabel as nib
    import trimesh
    from collections import defaultdict
    
    print("=" * 70)
    print("STAGE 6: Reprojecting boundary to original GIfTI surface")
    print("=" * 70)
    
    # Load original surface
    gii = nib.load(input_gii)
    orig_verts = gii.darrays[0].data.astype(np.float64)
    orig_faces = gii.darrays[1].data
    orig_mesh = trimesh.Trimesh(vertices=orig_verts, faces=orig_faces)
    
    # Find boundary nodes
    face_count = defaultdict(int)
    for tet in tets:
        for combo in [(tet[0],tet[1],tet[2]), (tet[0],tet[1],tet[3]),
                      (tet[0],tet[2],tet[3]), (tet[1],tet[2],tet[3])]:
            face_count[tuple(sorted(combo))] += 1
    
    boundary_nodes = set()
    for face, count in face_count.items():
        if count == 1:
            boundary_nodes.update(face)
    
    boundary_list = sorted(boundary_nodes)
    n_boundary = len(boundary_list)
    n_interior = points.shape[0] - n_boundary
    print(f"  Boundary nodes: {n_boundary}, Interior nodes: {n_interior}")
    
    # Measure drift before reprojection
    boundary_pts = points[boundary_list]
    closest_pts, distances, _ = orig_mesh.nearest.on_surface(boundary_pts)
    print(f"  Drift before: mean={distances.mean():.4f}mm, max={distances.max():.4f}mm")
    
    # Apply reprojection
    points_fixed = points.copy()
    for i, node_id in enumerate(boundary_list):
        points_fixed[node_id] = closest_pts[i]
    
    # Check for inverted tets
    n_inverted = 0
    for tet in tets:
        p = points_fixed[tet]
        vol = np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))
        if vol <= 0:
            n_inverted += 1
    
    print(f"  Inverted tets after reprojection: {n_inverted}")
    
    if n_inverted == 0:
        # Measure remaining drift (from original verts to nearest mesh node)
        from scipy.spatial import cKDTree
        tree = cKDTree(points_fixed)
        dists_after, _ = tree.query(orig_verts)
        print(f"  Surface distance after: mean={dists_after.mean():.4f}mm, max={dists_after.max():.4f}mm")
        print(f"  Reprojection successful.")
    else:
        print(f"  WARNING: {n_inverted} inverted tets. Reprojection may be too aggressive.")
        print(f"  Proceeding anyway — meshtool clean can fix small inversions.")
    
    print()
    return points_fixed, n_inverted


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(input_gii, output_mesh, cell_size=0.6, meshtool_path=None,
                 skip_meshtool=False, skip_gmsh=False, gmsh_passes=5,
                 reproject=False, work_dir=None, keep_intermediates=False):
    """
    Full pipeline: GIfTI → pygalmesh+exude → Gmsh Relocate3D → meshtool 3-pass → BrainGrowth .mesh.
    
    Parameters
    ----------
    input_gii : str
        Input GIfTI surface file.
    output_mesh : str
        Output BrainGrowth .mesh file.
    cell_size : float
        pygalmesh cell size in mm (0.6 default = the paper's default
        resolution; yields ~0.3-2.1M tets across the fetal age range).
    meshtool_path : str or None
        Path to meshtool binary. Default: ~/meshtool/meshtool.
    skip_meshtool : bool
        If True, skip meshtool cleaning.
    skip_gmsh : bool
        If True, skip Gmsh Relocate3D optimization.
    gmsh_passes : int
        Number of Gmsh Relocate3D passes (default: 5).
    reproject : bool
        If True, reproject boundary vertices onto original GIfTI surface.
        Reduces boundary drift (~0.33mm→0) at slight quality cost.
    work_dir : str or None
        Working directory for intermediate files.
    keep_intermediates : bool
        If True, keep intermediate CARP files.
    """
    
    if meshtool_path is None:
        meshtool_path = os.path.expanduser("~/meshtool/meshtool")
    
    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(output_mesh))
        if not os.access(work_dir, os.W_OK):
            work_dir = os.path.expanduser("~")
    
    output_base = os.path.splitext(os.path.basename(output_mesh))[0]
    carp_base = os.path.join(work_dir, f"{output_base}_carp_tmp")
    carp_gmsh = os.path.join(work_dir, f"{output_base}_carp_gmsh")
    carp_opt = os.path.join(work_dir, f"{output_base}_carp_opt")
    
    pipeline_start = time.time()
    
    print(f"\nInput:  {input_gii}")
    print(f"Output: {output_mesh}")
    print(f"Cell size: {cell_size}")
    print(f"Gmsh: {'SKIP' if skip_gmsh else f'{gmsh_passes} Relocate3D passes'}")
    print(f"Meshtool: {'SKIP' if skip_meshtool else meshtool_path}")
    print(f"Reproject: {'YES' if reproject else 'NO'}")
    print()
    
    # --- Stage 1: pygalmesh generation + exude ---
    points, tets = stage1_generate_volume_mesh(input_gii, cell_size, work_dir)
    
    if skip_meshtool and skip_gmsh:
        # Skip all optimization, write directly
        print("Skipping all optimization (--skip-meshtool --skip-gmsh).")
        write_braingrowth_mesh(points, tets, output_mesh)
        elapsed = time.time() - pipeline_start
        print(f"\nDone in {elapsed:.1f}s. Output: {output_mesh}")
        print(f"  Nodes: {points.shape[0]}, Tets: {tets.shape[0]}")
        return
    
    # --- Stage 2: Convert to CARP ---
    print("=" * 70)
    print("STAGE 2: Converting to CARP format")
    print("=" * 70)
    write_carp(points, tets, carp_base)
    print(f"  CARP files: {carp_base}.{{pts,elem,lon}}")
    
    # Query initial quality
    if not skip_meshtool:
        mean0, max0, std0 = query_meshtool_quality(carp_base, meshtool_path)
        if mean0 is not None:
            print(f"  Initial quality: mean={mean0:.4f}, max={max0:.4f}, stddev={std0:.4f}")
    print()
    
    # Track which CARP basename to feed into the next stage
    current_carp = carp_base
    
    # --- Stage 3: Gmsh Relocate3D (optional) ---
    if not skip_gmsh:
        gmsh_ok = stage3_gmsh_relocate(current_carp, carp_gmsh, n_passes=gmsh_passes)
        if gmsh_ok:
            current_carp = carp_gmsh
    
    # --- Stage 4: Iterative meshtool cleaning ---
    if not skip_meshtool:
        stage4_meshtool_clean(current_carp, carp_opt, meshtool_path, thresholds=[0.7, 0.5, 0.3])
        current_carp = carp_opt
    
    # Query final quality
    final_mean = final_max = final_std = None
    if not skip_meshtool:
        final_mean, final_max, final_std = query_meshtool_quality(current_carp, meshtool_path)
        if final_mean is not None:
            print(f"  Final quality: mean={final_mean:.4f}, max={final_max:.4f}, stddev={final_std:.4f}")
    
    # --- Stage 5: Convert to BrainGrowth format ---
    print("=" * 70)
    print("STAGE 5: Converting to BrainGrowth .mesh format")
    print("=" * 70)
    
    final_points, final_tets = read_carp(current_carp)
    
    # --- Stage 6 (optional): Reproject boundary ---
    if reproject:
        final_points, n_inverted = stage6_reproject_boundary(input_gii, final_points, final_tets)
        if n_inverted > 0 and not skip_meshtool:
            # Run a light meshtool pass on the reprojected mesh to fix inversions
            print("  Running repair pass on reprojected mesh...")
            carp_reproj = os.path.join(work_dir, f"{output_base}_carp_reproj")
            carp_reproj_clean = os.path.join(work_dir, f"{output_base}_carp_reproj_clean")
            write_carp(final_points, final_tets, carp_reproj)
            stage4_meshtool_clean(carp_reproj, carp_reproj_clean, meshtool_path, thresholds=[0.5])
            final_points, final_tets = read_carp(carp_reproj_clean)
            # Cleanup repair files
            if not keep_intermediates:
                for ext in [".pts", ".elem", ".lon", ".fcon"]:
                    for base in [carp_reproj, carp_reproj_clean]:
                        fpath = f"{base}{ext}"
                        if os.path.exists(fpath):
                            os.remove(fpath)
    
    write_braingrowth_mesh(final_points, final_tets, output_mesh)
    
    print(f"  Output: {output_mesh}")
    print(f"  Nodes: {final_points.shape[0]}, Tets: {final_tets.shape[0]}")
    
    # Verify output format
    with open(output_mesh) as f:
        first_line = f.readline().strip()
    print(f"  Format check: first line = '{first_line}' (node count)")
    
    # --- Cleanup intermediates ---
    if not keep_intermediates:
        for ext in [".pts", ".elem", ".lon", ".fcon"]:
            for base in [carp_base, carp_gmsh, carp_opt,
                         f"{carp_opt}_pass1", f"{carp_opt}_pass2"]:
                fpath = f"{base}{ext}"
                if os.path.exists(fpath):
                    os.remove(fpath)
        print("  Intermediate files cleaned up.")
    else:
        print(f"  Intermediate CARP files kept in: {work_dir}")
    
    # --- Summary ---
    elapsed = time.time() - pipeline_start
    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output: {output_mesh}")
    print(f"  Nodes: {final_points.shape[0]}")
    print(f"  Tetrahedra: {final_tets.shape[0]}")
    if final_mean is not None:
        print(f"  Quality (tet_qmetric_volume, 0=best 1=worst):")
        print(f"    Mean:   {final_mean:.4f}")
        print(f"    Max:    {final_max:.4f}")
        print(f"    Stddev: {final_std:.4f}")
        print(f"  Reference (Tallinen 22W): mean=0.162, max=0.811")
    print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GIfTI surface → optimized BrainGrowth tetrahedral .mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run, at the paper's default resolution h = 0.6 mm
  python3 %(prog)s -i brain_surface.gii -o brain_volume.mesh

  # Finer mesh (~1M tets)
  python3 %(prog)s -i brain_surface.gii -o brain_volume.mesh --cell-size 0.5

  # Quick test without meshtool
  python3 %(prog)s -i brain_surface.gii -o brain_volume.mesh --skip-meshtool

On a module-based HPC system, pygalmesh needs eigen, gmp and boost loaded
(do NOT load a cgal module -- pygalmesh ships its own headers). Exact module
names are site-specific; see docs/ENVIRONMENT.md.
        """,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input GIfTI (.gii) surface file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output BrainGrowth .mesh file")
    parser.add_argument("--cell-size", type=float, default=0.6,
                        help="Cell size h in mm: the upper bound on each tet's "
                             "circumradius and on the surface Delaunay ball "
                             "radius, with facet distance h/2 "
                             "(default: 0.6, the paper's default resolution)"),
    parser.add_argument("--meshtool-path", type=str, default=None,
                        help="Path to meshtool binary (default: ~/meshtool/meshtool)")
    parser.add_argument("--skip-meshtool", action="store_true",
                        help="Skip meshtool optimization passes")
    parser.add_argument("--skip-gmsh", action="store_true",
                        help="Skip Gmsh Relocate3D optimization")
    parser.add_argument("--gmsh-passes", type=int, default=5,
                        help="Number of Gmsh Relocate3D passes (default: 5)")
    parser.add_argument("--reproject", action="store_true",
                        help="Reproject boundary vertices onto original GIfTI surface (reduces drift, slight quality cost)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate CARP files")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for temp files (default: output dir)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if not args.skip_meshtool:
        mt = args.meshtool_path or os.path.expanduser("~/meshtool/meshtool")
        if not os.path.exists(mt):
            print(f"ERROR: meshtool not found at {mt}", file=sys.stderr)
            print("Use --meshtool-path or --skip-meshtool", file=sys.stderr)
            sys.exit(1)
    
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    
    run_pipeline(
        input_gii=args.input,
        output_mesh=args.output,
        cell_size=args.cell_size,
        meshtool_path=args.meshtool_path,
        skip_meshtool=args.skip_meshtool,
        skip_gmsh=args.skip_gmsh,
        gmsh_passes=args.gmsh_passes,
        reproject=args.reproject,
        work_dir=args.work_dir,
        keep_intermediates=args.keep_intermediates,
    )
