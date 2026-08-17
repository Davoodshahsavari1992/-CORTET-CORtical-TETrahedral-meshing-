#!/usr/bin/env python3
r"""check_node_reordering.py -- confirm surface-first node ordering.

Answers one decidable question about any .mesh file: are its boundary nodes
genuinely numbered first, indices 0..N_surf-1, as the solver requires?
Modifies nothing. Run it on any mesh that reached your solver by a route
other than finalize_for_braingrowth.py.

    python3 check_node_reordering.py --mesh subject.mesh
    python3 check_node_reordering.py --selftest
"""
import argparse
from collections import Counter

import numpy as np


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


def boundary_vertex_ids(tets):
    face_count = Counter()
    for tet in tets:
        a, b, c, d = tet
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_count[tuple(sorted(face))] += 1
    return sorted(set(v for f, cnt in face_count.items() if cnt == 1 for v in f))


def check(nodes, tets):
    """Returns a dict report. is_surface_first is the key result."""
    boundary_ids = boundary_vertex_ids(tets)
    n_surf = len(boundary_ids)
    n_total = len(nodes)
    expected = list(range(n_surf))
    is_surface_first = (boundary_ids == expected)

    # if it's NOT surface-first, characterise how bad the mismatch is,
    # since "off by a little" vs "completely scrambled" both fail the
    # check but mean different things for how urgently this needs fixing
    if not is_surface_first:
        boundary_set = set(boundary_ids)
        expected_set = set(expected)
        wrongly_included = len(expected_set - boundary_set)  # in [0,n_surf) but interior
        wrongly_excluded = len(boundary_set - expected_set)  # boundary but outside [0,n_surf)
    else:
        wrongly_included = wrongly_excluded = 0

    return {
        "n_nodes": n_total,
        "n_boundary": n_surf,
        "n_interior": n_total - n_surf,
        "is_surface_first": is_surface_first,
        "wrongly_included_in_range": wrongly_included,
        "wrongly_excluded_from_range": wrongly_excluded,
    }


def print_report(report, mesh_path=None):
    if mesh_path:
        print(f"Mesh: {mesh_path}")
    print(f"  Total nodes:    {report['n_nodes']}")
    print(f"  Boundary nodes: {report['n_boundary']}")
    print(f"  Interior nodes: {report['n_interior']}")
    print()
    if report["is_surface_first"]:
        print("  RESULT: PASS -- boundary nodes are exactly indices "
              f"[0, {report['n_boundary']-1}]. Surface-first ordering holds.")
        print("  Safe for the solver's tetra_normals() indexing.")
    else:
        print("  RESULT: FAIL -- boundary nodes are NOT exactly the first")
        print(f"  {report['n_boundary']} indices.")
        print(f"    Interior nodes wrongly sitting inside [0, N_surf): "
              f"{report['wrongly_included_in_range']}")
        print(f"    Boundary nodes wrongly sitting outside [0, N_surf): "
              f"{report['wrongly_excluded_from_range']}")
        print()
        print("  This mesh's surface-normal indexing in the solver is NOT")
        print("  reliable. Any simulation already run on this exact file may")
        print("  have used wrong surface normals silently (no crash expected).")
        print("  Re-run finalize_for_braingrowth.py on this mesh's source")
        print("  points/tets to produce a correctly-ordered file before")
        print("  trusting simulation results from it.")


def selftest():
    # ---- Case A: a mesh DELIBERATELY built surface-first (should PASS) ----
    # single tetrahedron: all 4 vertices are boundary vertices (no interior
    # in a single tet) -- trivially surface-first since boundary IS [0,3]
    pts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tets_a = np.array([[0, 1, 2, 3]])
    report_a = check(pts_a, tets_a)
    assert report_a["is_surface_first"], \
        f"Case A should PASS (trivially surface-first), got {report_a}"
    assert report_a["n_boundary"] == 4 and report_a["n_interior"] == 0

    # ---- Case B: a mesh with a genuine interior vertex, correctly ordered
    # (surface nodes 0-5, interior node 6) -- should PASS ----
    outer = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    interior = np.array([[0, 0, 0]], dtype=float)
    pts_b = np.vstack([outer, interior])  # interior is index 6, AFTER surface -- correct
    tets_b = []
    for a, b in [(0, 2), (2, 1), (1, 3), (3, 0)]:
        tets_b.append([a, b, 4, 6])
        tets_b.append([a, b, 5, 6])
    tets_b = np.array(tets_b)
    report_b = check(pts_b, tets_b)
    assert report_b["is_surface_first"], \
        f"Case B should PASS (interior node correctly placed last), got {report_b}"
    assert report_b["n_boundary"] == 6 and report_b["n_interior"] == 1

    # ---- Case C: the SAME geometry as Case B, but with node order
    # DELIBERATELY scrambled so the interior node is NOT last -- should FAIL ----
    # put interior node FIRST instead of last, and remap tets to match
    pts_c = np.vstack([interior, outer])  # interior is now index 0 -- WRONG
    remap = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # old->new after moving interior to front... 
    # simpler: just directly build tets_c referencing the new indexing
    # (interior=0, outer points are now 1..6)
    tets_c = []
    for a, b in [(1, 3), (3, 2), (2, 4), (4, 1)]:  # outer indices shifted +1
        tets_c.append([a, b, 5, 0])   # 0 = interior (WRONG position), 5 = one pole (+1 shift)
        tets_c.append([a, b, 6, 0])   # 6 = other pole (+1 shift)
    tets_c = np.array(tets_c)
    report_c = check(pts_c, tets_c)
    assert not report_c["is_surface_first"], \
        f"Case C should FAIL (interior node deliberately placed first), got {report_c}"
    assert report_c["wrongly_included_in_range"] > 0 or report_c["wrongly_excluded_from_range"] > 0

    print("selftest PASS:")
    print("  Case A (single tet, trivially surface-first): correctly PASSES")
    print("  Case B (interior node correctly placed last):  correctly PASSES")
    print("  Case C (interior node deliberately placed first, same")
    print("          geometry as Case B): correctly FAILS")
    print("  -- confirms this script actually distinguishes correct from")
    print("     incorrect node ordering, not just reporting one answer.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.mesh:
        ap.error("--mesh is required (or --selftest)")

    nodes, tets = read_mesh(args.mesh)
    report = check(nodes, tets)
    print_report(report, args.mesh)


if __name__ == "__main__":
    main()
