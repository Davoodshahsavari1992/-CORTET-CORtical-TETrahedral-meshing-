#!/usr/bin/env python3
r"""smooth_one_multi_N.py -- pre-mesh surface smoothing sweep (paper Section 3.5).

Smooths one cortical surface at several iteration counts with a
surface-preserving Laplacian filter, writing one surface per count plus a
metrics CSV, so the sweep can be inspected together in wb_view before one N
is committed to.

How much smoothing a surface tolerates falls sharply with gestational age: a
young, near-smooth cortex is mostly noise, an older folded one deflates fast
and its curvature is real anatomy.

    python3 smooth_one_multi_N.py --surface subject_left_white.surf.gii \
        --out-dir smoothing/ --N 0 10 20 30 50
    python3 smooth_one_multi_N.py --selftest
"""
import argparse, csv, sys
from pathlib import Path

# Imports are guarded so that a missing dependency produces a readable message
# (and --selftest still runs to tell you exactly what is missing) rather than a
# bare traceback.
_MISSING = []
try:
    import numpy as np
except ImportError:
    np = None
    _MISSING.append("numpy")
try:
    import nibabel as nib
except ImportError:
    nib = None
    _MISSING.append("nibabel")


def _report_missing():
    print("Missing required package(s): " + ", ".join(_MISSING))
    print("\nInstall them, then re-run --selftest:")
    if "numpy" in _MISSING or "nibabel" in _MISSING:
        print("    pip install --user numpy nibabel scipy")
    print("    pip install --user pymeshlab --no-deps"
          "        # --no-deps stops pip upgrading numpy")
    return 1


# ---------------------------------------------------------------- I/O

def load_surf(p):
    g = nib.load(str(p))
    return (g.agg_data("pointset").astype(np.float64),
            g.agg_data("triangle").astype(np.int64))


def save_surf(v, f, out_path, like=None):
    """Write a .surf.gii, carrying over the original's metadata/coordsys so
    downstream tools (wb_command, meshers) see the same structure."""
    va = nib.gifti.GiftiDataArray(np.asarray(v, np.float32),
                                  intent="NIFTI_INTENT_POINTSET",
                                  datatype="NIFTI_TYPE_FLOAT32")
    fa = nib.gifti.GiftiDataArray(np.asarray(f, np.int32),
                                  intent="NIFTI_INTENT_TRIANGLE",
                                  datatype="NIFTI_TYPE_INT32")
    img = nib.gifti.GiftiImage(darrays=[va, fa])
    if like is not None:
        try:
            orig = nib.load(str(like))
            img.meta = orig.meta
            for dn, do in zip(img.darrays, orig.darrays):
                dn.meta = do.meta
                dn.coordsys = do.coordsys
        except Exception:
            pass
    nib.save(img, str(out_path))


def derive_stem(name):
    """Strip known surface suffixes without mangling unfamiliar filenames.

    The naive `name.replace('_white.surf.gii','')` silently does nothing when the
    file is named differently, producing outputs like
    'sub-X_lh.surf.gii_white_sp30.surf.gii'. This handles any naming."""
    for suf in ("_white.surf.gii", ".surf.gii", ".gii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


# ---------------------------------------------------------------- smoothing

def surface_preserving(v, f, iters, angle=60.0):
    """surface-preserving Laplacian via pymeshlab. N=0 returns the raw surface.

    The keyword names for this filter changed between pymeshlab versions, so we
    try each known spelling rather than pinning a version."""
    if int(iters) == 0:
        return v.copy(), f.copy()
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=v, face_matrix=f))
    last = None
    for kw in ({"angledeg": float(angle), "iterations": int(iters)},
               {"anglethreshold": float(angle), "iterations": int(iters)},
               {"angledeg": float(angle), "stepsmoothnum": int(iters)}):
        try:
            ms.apply_coord_laplacian_smoothing_surface_preserving(**kw)
            m = ms.current_mesh()
            return (m.vertex_matrix().astype(np.float64),
                    m.face_matrix().astype(np.int64))
        except TypeError as e:
            last = e
            continue
    raise RuntimeError(
        "pymeshlab's surface_preserving filter did not accept any known "
        f"parameter spelling (last error: {last}). Check your pymeshlab version "
        "and the filter signature with: "
        "pymeshlab.print_filter_parameter_list('apply_coord_laplacian_smoothing_surface_preserving')")


# ---------------------------------------------------------------- metrics

def tri_area(v, f):
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def _adjacency(n, faces):
    from scipy.sparse import coo_matrix
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]], axis=0)
    A = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])),
                   shape=(n, n)).tocsr()
    A.data = np.minimum(A.data, 1.0)
    return A


def roughness(v, A):
    """Mean distance of each vertex from the average of its neighbours
    (umbrella-Laplacian). Higher = rougher."""
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return float(np.linalg.norm(v - (A @ v) / deg[:, None], axis=1).mean())


def chamfer(v0, v1):
    from scipy.spatial import cKDTree
    d01, _ = cKDTree(v1).query(v0)
    d10, _ = cKDTree(v0).query(v1)
    return float(0.5 * (d01.mean() + d10.mean()))


# ---------------------------------------------------------------- selftest

def selftest(angle=60.0):
    """Confirm pymeshlab is installed, importable, and the filter runs. Uses a
    tiny built-in mesh - no external data needed."""
    print("checking imports...")
    if _MISSING:
        return _report_missing()
    try:
        import pymeshlab  # noqa: F401
    except ImportError as e:
        print(f"  FAIL: pymeshlab not importable ({e})")
        print("  install with:  pip install --user pymeshlab --no-deps")
        print("  (the --no-deps stops pip upgrading numpy and breaking other tools)")
        return 1
    try:
        import scipy  # noqa: F401
        has_scipy = True
    except ImportError:
        has_scipy = False
    print(f"  numpy {np.__version__}, nibabel {nib.__version__}, "
          f"scipy {'ok' if has_scipy else 'MISSING (metrics will be skipped)'}")
    if not np.__version__.startswith("1.2"):
        print(f"  NOTE: numpy is {np.__version__}. If you share this environment "
              "with a mesh pipeline pinned to numpy 1.24.x, check it still works.")

    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], float)
    f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]], int)
    print("running surface_preserving at N=5 on a 5-vertex test mesh...")
    v2, f2 = surface_preserving(v, f, 5, angle)
    assert v2.shape == v.shape, f"vertex count changed: {v.shape} -> {v2.shape}"
    assert f2.shape == f.shape, f"face count changed: {f.shape} -> {f2.shape}"
    print("  OK: filter ran, topology preserved.")
    print("\nselftest PASSED - you can run this on real surfaces.")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", help="input surface (.surf.gii)")
    ap.add_argument("--out-dir", help="folder for the smoothed surfaces")
    ap.add_argument("--N", type=int, nargs="+", default=[0, 10, 20, 30, 40],
                    help="iteration counts to generate (0 = copy of raw)")
    ap.add_argument("--angle", type=float, default=60.0,
                    help="dihedral angle threshold; inert above ~20, leave at 60")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate files that already exist")
    ap.add_argument("--selftest", action="store_true",
                    help="check the install works, then exit (no data needed)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.angle)

    if _MISSING:
        return _report_missing()

    if not args.surface or not args.out_dir:
        ap.error("--surface and --out-dir are required (or use --selftest)")

    surf = Path(args.surface).expanduser()
    if not surf.is_file():
        print(f"ERROR: input surface not found: {surf}")
        return 1
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    try:
        import scipy  # noqa: F401
        metrics_on = True
    except ImportError:
        metrics_on = False
        print("NOTE: scipy not found - area will still be reported, but "
              "noise-removed and chamfer will be skipped.\n")

    v0, f0 = load_surf(surf)
    a0 = tri_area(v0, f0)
    stem = derive_stem(surf.name)
    A = _adjacency(len(v0), f0) if metrics_on else None
    r0 = roughness(v0, A) if metrics_on else None

    print(f"input   : {surf.name}")
    print(f"vertices: {len(v0)}   faces: {len(f0)}   area: {a0:.1f}")
    if metrics_on:
        print(f"roughness (raw): {r0:.5f}")
    print(f"smoothing at N = {sorted(set(args.N))}\n")

    if metrics_on:
        print(f"{'N':>5} {'noise_removed%':>15} {'area%':>9} {'chamfer_mm':>12}   file")
    else:
        print(f"{'N':>5} {'area%':>9}   file")

    rows = []
    for n in sorted(set(args.N)):
        op = out / f"{stem}_sp{n}.surf.gii"
        if op.is_file() and not args.overwrite:
            print(f"{n:>5}   (exists, skipped - use --overwrite)   {op.name}")
            continue
        vs, fs = surface_preserving(v0, f0, n, args.angle)
        if vs.shape != v0.shape:
            print(f"  WARNING at N={n}: vertex count changed "
                  f"{v0.shape[0]} -> {vs.shape[0]}. Not expected for this filter.")
        save_surf(vs, fs, op, like=surf)
        if not op.is_file():
            print(f"  ERROR: failed to write {op}")
            continue

        area = 100.0 * (tri_area(vs, fs) - a0) / a0
        row = {"N": n, "area_pct": round(area, 3), "file": op.name}
        if metrics_on:
            noise = 100.0 * (1 - roughness(vs, A) / r0) if r0 > 0 else 0.0
            cham = chamfer(v0, vs)
            row["noise_removed_pct"] = round(noise, 2)
            row["chamfer_mm"] = round(cham, 4)
            flag = ""
            if noise < 0:
                flag += "  <- NEGATIVE: adding roughness"
            if area < -6:
                flag += "  [area loss > 6%]"
            if cham > 0.25:
                flag += "  [drift > 0.25mm]"
            print(f"{n:>5} {noise:>15.1f} {area:>9.2f} {cham:>12.4f}   {op.name}{flag}")
        else:
            print(f"{n:>5} {area:>9.2f}   {op.name}")
        rows.append(row)

    if rows:
        cols = (["N", "noise_removed_pct", "area_pct", "chamfer_mm", "file"]
                if metrics_on else ["N", "area_pct", "file"])
        mpath = out / f"{stem}_smoothing_metrics.csv"
        with open(mpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nmetrics -> {mpath}")

    print(f"\nnow look at them together and decide:")
    print(f"  wb_view {out}/{stem}_sp*.surf.gii")
    print("compare each against the N=0 (raw) copy in the same folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
