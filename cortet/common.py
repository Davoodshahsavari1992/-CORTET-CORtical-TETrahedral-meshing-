#!/usr/bin/env python3
r"""common.py -- the shared core the stage runners in run/ are built on.

Mesh I/O, the orientation convention, the meshtool quality measurement and the
console formatting live here once, so every stage reports numbers that are
directly comparable.

    python3 cortet/common.py --selftest
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import numpy as np

# tet_qmetric_volume: 0 = regular tetrahedron, 1 = degenerate.
SOLVER_READY_THRESHOLD = 0.6

# The paper's default resolution. Every cohort number was produced at this h.
DEFAULT_CELL_SIZE = 0.6

# Generation-stage cleaning. Post-mesh smoothing uses [0.7, 0.5] instead --
# a third pass at 0.3 fails to converge there.
GENERATION_THRESHOLDS = [0.7, 0.5, 0.3]
POSTMESH_THRESHOLDS = [0.7, 0.5]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Console reporting. Kept deliberately plain: these scripts are meant to be run
# under SLURM, where the log file is the only record of what happened.
# ---------------------------------------------------------------------------

def banner(title, char="="):
    print(f"\n{char * 74}\n{title}\n{char * 74}", flush=True)


def step(msg):
    print(f"  {msg}", flush=True)


def warn(msg):
    print(f"  WARNING: {msg}", flush=True)


def fail(msg, code=1):
    print(f"\nFAILED: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def report_quality(label, q, n_nodes=None, n_tets=None):
    """One consistent line per measured configuration."""
    if not q:
        print(f"  {label:<28} MEASUREMENT FAILED", flush=True)
        return
    size = ""
    if n_nodes is not None and n_tets is not None:
        size = f"  nodes {n_nodes:>9,}  tets {n_tets:>10,}"
    over = q.get("n_above")
    over_s = f"  q>{SOLVER_READY_THRESHOLD} {over:>9,}" if over is not None else ""
    verdict = ""
    if q.get("max") is not None:
        verdict = "  PASS" if q["max"] < SOLVER_READY_THRESHOLD else "  FAIL"
    print(f"  {label:<28}{size}  qbar {q.get('mean', float('nan')):.4f}"
          f"  qmax {q.get('max', float('nan')):.4f}{over_s}{verdict}", flush=True)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def read_braingrowth_mesh(path, want_faces=False):
    """
    Read the three-section .mesh format the pipeline writes.

        <n_nodes> / x y z ...
        <n_tets>  / tag n1 n2 n3 n4   (1-based)
        <n_faces> / tag f1 f2 f3      (1-based, optional third section)

    FIVE fields per tet line, not four. The leading tag column is the trap: a
    reader that takes four integers ingests the tag as node 1, drops node 4, and
    silently corrupts every element. This project shipped that bug once.

    Returns (points, tets) or (points, tets, faces) if want_faces.
    All indices 0-based. `faces` is None when the file has no third section.
    """
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    n_nodes = int(lines[i].split()[0]); i += 1
    points = np.empty((n_nodes, 3), dtype=np.float64)
    for k in range(n_nodes):
        p = lines[i].split(); i += 1
        points[k] = (float(p[0]), float(p[1]), float(p[2]))

    n_tets = int(lines[i].split()[0]); i += 1
    tets = np.empty((n_tets, 4), dtype=np.int64)
    for k in range(n_tets):
        p = lines[i].split(); i += 1
        if len(p) < 5:
            raise ValueError(
                f"{path} line {i}: {len(p)} fields on a tet line, expected 5 "
                f"(tag n1 n2 n3 n4). This is not the format this pipeline writes.")
        tets[k] = [int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1, int(p[4]) - 1]

    faces = None
    if i < len(lines) and lines[i].strip():
        n_faces = int(lines[i].split()[0]); i += 1
        faces = np.empty((n_faces, 3), dtype=np.int64)
        for k in range(n_faces):
            p = lines[i].split(); i += 1
            faces[k] = [int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1]

    return (points, tets, faces) if want_faces else (points, tets)


def read_gii_surface(path):
    """Load a GIfTI surface. Returns (vertices, faces) as float64 / int64."""
    import nibabel as nib
    g = nib.load(path)
    return (np.asarray(g.darrays[0].data, dtype=np.float64),
            np.asarray(g.darrays[1].data, dtype=np.int64))


def write_carp(points, tets, basename):
    """Write CARP text format (.pts/.elem/.lon), which is what meshtool reads."""
    with open(f"{basename}.pts", "w") as fh:
        fh.write(f"{len(points)}\n")
        fh.writelines(f"{p[0]} {p[1]} {p[2]}\n" for p in points)
    with open(f"{basename}.elem", "w") as fh:
        fh.write(f"{len(tets)}\n")
        fh.writelines("Tt " + " ".join(map(str, t)) + " 1\n" for t in tets)
    with open(f"{basename}.lon", "w") as fh:            # meshtool requires it
        fh.write("1\n")
        fh.writelines("1.0 0.0 0.0\n" for _ in tets)


def remove_carp(basename):
    for ext in (".pts", ".elem", ".lon", ".fcon"):
        try:
            os.remove(basename + ext)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def signed_volumes(points, tets):
    """Six times the signed volume of each tet. Negative means inverted."""
    p = points[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                     np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))


def orient_positive(points, tets):
    """
    Return a copy of `tets` with every element positively oriented.

    This is REQUIRED before measuring. The solver format deliberately stores
    tets in a negative-volume convention, and a
    quality tool handed that convention reports the mesh as broken while both
    tools are behaving correctly. Measure in the standard convention, always.
    """
    tets = np.array(tets, dtype=np.int64, copy=True)
    neg = signed_volumes(points, tets) < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return tets, int(neg.sum())


# ---------------------------------------------------------------------------
# meshtool
# ---------------------------------------------------------------------------

def _parse_quality_line(stdout):
    """meshtool prints: 'Min: <v> Max: <v> Mean: <v> Stddev: <v>'."""
    q = {}
    for line in stdout.split("\n"):
        if line.strip().startswith("Min:"):
            parts = line.split()
            for j, tok in enumerate(parts):
                if tok in ("Min:", "Max:", "Mean:", "Stddev:"):
                    try:
                        q[tok.strip(":").lower()] = float(parts[j + 1])
                    except (IndexError, ValueError):
                        pass
    return q


def measure_mesh(points, tets, meshtool, work_dir, per_element=False,
                 threshold=SOLVER_READY_THRESHOLD, tag="q"):
    """
    Measure a mesh with meshtool's tet_qmetric_volume — the measure the paper
    reports and the measure Stage 4 repairs against.

    Reorients to positive volume first (see orient_positive).

    With per_element=True it also asks for the per-element dump (`-odat`), which
    is the only way to get the count of elements above the threshold; meshtool's
    summary line gives min/max/mean/stddev but no count. meshtool appends
    `.qual.dat` to whatever `-odat` path you give it.

    Returns a dict with min/max/mean/stddev, plus n_above and `values` when
    per_element is on. Returns {} if meshtool produced nothing parseable.
    """
    os.makedirs(work_dir, exist_ok=True)
    tets, n_flipped = orient_positive(points, tets)
    base = os.path.join(work_dir, f"_measure_{tag}")
    write_carp(points, tets, base)

    cmd = [meshtool, "query", "quality", f"-msh={base}", "-ifmt=carp_txt"]
    odat = base + "_pe.dat"
    if per_element:
        cmd.append(f"-odat={odat}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, PermissionError):
        remove_carp(base)
        warn(f"meshtool not found at '{meshtool}'. The mesh was written; only "
             f"the quality measurement is unavailable. Pass --meshtool with the "
             f"path to the binary, and see the README's Installation section.")
        return {}
    q = _parse_quality_line(result.stdout)
    if q:
        q["n_flipped_for_measurement"] = n_flipped

    # tet_qmetric_volume is normalised to [0, 1]. A value outside that range is
    # not a bad mesh but a MALFORMED one: the measure itself has broken down.
    # TetGen run without PyVista's normal-consistency prep returns qmax = 2.0.
    # The tolerance is needed because a perfect element legitimately returns
    # something like -1e-17, which is negative zero rather than a bad mesh.
    EPS = 1e-6
    for key in ("min", "max", "mean"):
        v = q.get(key)
        if v is not None and not (-EPS <= v <= 1.0 + EPS):
            warn(f"q_{key} = {v:.4f} is OUTSIDE the metric's [0,1] range. The "
                 f"input is malformed, not merely poor. Check surface "
                 f"orientation and preparation before reading anything else "
                 f"into this number.")
            q["out_of_range"] = True

    if per_element and q:
        candidates = [odat + ".qual.dat", odat]
        candidates += sorted(glob.glob(os.path.join(work_dir, "*.qual.dat")))
        found = next((c for c in candidates
                      if os.path.exists(c) and os.path.getsize(c) > 0), None)
        if found:
            values = np.loadtxt(found, dtype=np.float64)
            q["values"] = values
            q["n_above"] = int((values > threshold).sum())
            try:
                os.remove(found)
            except OSError:
                pass
        else:
            warn("meshtool -odat produced no per-element file; "
                 "the count of elements above threshold is unavailable")

    remove_carp(base)
    if not q:
        warn(f"meshtool returned nothing parseable. stderr: "
             f"{result.stderr.strip()[:200]}")
    return q


def measure_mesh_file(path, meshtool, work_dir, **kw):
    """measure_mesh() straight from a .mesh file."""
    points, tets = read_braingrowth_mesh(path)
    q = measure_mesh(points, tets, meshtool, work_dir, **kw)
    return q, len(points), len(tets)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def find_meshtool(explicit=None):
    """Resolve the meshtool binary: explicit flag, then $PATH, then ~/meshtool."""
    for candidate in (explicit, shutil.which("meshtool"),
                      os.path.expanduser("~/meshtool/meshtool")):
        if candidate and os.path.exists(candidate):
            return candidate
    return explicit or "meshtool"


def meshtool_problem(meshtool, detail):
    """The message to show when check_meshtool_works() says no. Distinguishes
    a missing binary from one that is present but returns the wrong answer --
    they need different fixes."""
    if detail == "binary not found":
        return (f"meshtool not found at '{meshtool}'. It is not pip-installable; "
                f"see the README's Installation section, and pass --meshtool with "
                f"the path to the binary.")
    return (f"meshtool is present but not measuring correctly ({detail}). "
            f"See docs/ENVIRONMENT.md -- this is almost always the module "
            f"environment rather than the mesh.")


def check_meshtool_works(meshtool, work_dir):
    """
    Confirm meshtool actually functions, by measuring a mesh whose answer we
    know: a single regular tetrahedron, which must score near 0.

    This exists because meshtool can exit 0 and emit empty or wrong output when
    run outside the correct module environment, with nothing pointing at the
    cause (docs/ENVIRONMENT.md). Checking the binary exists is not enough.

    Returns (ok: bool, detail: str).
    """
    if not os.path.exists(meshtool) and not shutil.which(meshtool):
        return False, "binary not found"

    a = np.sqrt(2.0) / 2.0
    points = np.array([[1.0, 0, -a], [-1, 0, -a], [0, 1, a], [0, -1, a]])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    try:
        q = measure_mesh(points, tets, meshtool, work_dir, tag="envcheck")
    except Exception as exc:                                  # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"

    if not q or q.get("max") is None:
        return False, "ran but produced no parseable quality output"
    if q["max"] > 0.2:
        return False, (f"a regular tetrahedron scored {q['max']:.4f}, expected "
                       f"near 0 — the environment is wrong, not the mesh")
    return True, f"regular tetrahedron scores {q['max']:.4f}"


def script_path(*parts):
    """Absolute path to a script in this repository, from anywhere."""
    return os.path.join(REPO_ROOT, *parts)


def run_subprocess(cmd, label, allow_fail=False):
    """Run a child script, echoing the command so the log records exactly what ran."""
    banner(label)
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0 and not allow_fail:
        fail(f"{label} exited {result.returncode}. If this is a generation or "
             f"meshtool step failing on missing libraries, see docs/ENVIRONMENT.md "
             f"— the HPC modules must be loaded in your shell first.",
             result.returncode)
    return result.returncode == 0


# ---------------------------------------------------------------------------

def _selftest():
    import tempfile

    banner("common.py selftest")

    a = np.sqrt(2.0) / 2.0
    pts = np.array([[1.0, 0, -a], [-1, 0, -a], [0, 1, a], [0, -1, a]])

    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    v = signed_volumes(pts, tets)[0]
    flipped, n = orient_positive(pts, tets)
    assert signed_volumes(pts, flipped)[0] > 0
    assert n == (1 if v < 0 else 0)
    step(f"orient_positive: signed volume {v:+.4f} -> "
         f"{signed_volumes(pts, flipped)[0]:+.4f}, flipped {n}          OK")

    # a mesh already positive must be left completely alone
    same, n0 = orient_positive(pts, flipped)
    assert n0 == 0 and np.array_equal(same, flipped)
    step("orient_positive is idempotent                                  OK")

    with tempfile.TemporaryDirectory() as d:
        mesh = os.path.join(d, "one.mesh")
        with open(mesh, "w") as fh:
            fh.write("4\n")
            for p in pts:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            fh.write("1\n   1         1         2         4         3\n")
            fh.write("1\n   1         1         2         3\n")
        p2, t2, f2 = read_braingrowth_mesh(mesh, want_faces=True)
        assert p2.shape == (4, 3) and t2.tolist() == [[0, 1, 3, 2]]
        assert f2.tolist() == [[0, 1, 2]]
        step("read_braingrowth_mesh: 5-field tet line, 3 sections          OK")

        bad = os.path.join(d, "bad.mesh")
        with open(bad, "w") as fh:
            fh.write("4\n")
            for p in pts:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            fh.write("1\n   1 2 4 3\n")
        try:
            read_braingrowth_mesh(bad)
        except ValueError:
            step("4-field tet line rejected, not silently misread             OK")
        else:
            raise AssertionError("4-field tet line was accepted")

        write_carp(pts, flipped, os.path.join(d, "c"))
        assert all(os.path.exists(os.path.join(d, "c" + e))
                   for e in (".pts", ".elem", ".lon"))
        with open(os.path.join(d, "c.elem")) as fh:
            assert fh.readline().strip() == "1"
            assert fh.readline().split()[0] == "Tt"
        step("write_carp: .pts/.elem/.lon written, .lon present            OK")
        remove_carp(os.path.join(d, "c"))
        assert not os.path.exists(os.path.join(d, "c.pts"))
        step("remove_carp cleans up                                        OK")

    q = _parse_quality_line("Min: 0.000568 Max: 0.997225 Mean: 0.244539 "
                            "Stddev: 0.161467")
    assert q == {"min": 0.000568, "max": 0.997225,
                 "mean": 0.244539, "stddev": 0.161467}
    step("meshtool summary line parsed                                  OK")
    assert _parse_quality_line("nothing useful here") == {}
    step("unparseable meshtool output yields {}, not a wrong number     OK")

    assert os.path.exists(script_path("cortet", "common.py"))
    step("script_path resolves inside the repository                    OK")

    print("\nselftest passed.\n")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
        print("This is a library. Run the numbered stages in run/ instead.")
