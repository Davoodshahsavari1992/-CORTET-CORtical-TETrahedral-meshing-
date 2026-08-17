#!/usr/bin/env python3
r"""compare_h_direct.py -- boundary roughness of the same surface at several h.

    python3 compare_h_direct.py --gii input.surf.gii \
        --meshes subject_h0.4.mesh subject_h0.6.mesh subject_h0.8.mesh
"""
import argparse
import numpy as np
from compare_roughness_gii_vs_mesh import (
    read_gii_surface, read_mesh, boundary_surface_from_mesh, roughness
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gii", required=True)
    ap.add_argument("--meshes", nargs="+", required=True,
                     help="label=path pairs, e.g. h0.6=file1.mesh h0.4=file2.mesh")
    args = ap.parse_args()

    gii_verts, gii_faces = read_gii_surface(args.gii)
    gii_r, gii_n = roughness(gii_verts, gii_faces)
    print(f"Input gii roughness: {gii_r:.5f}  ({gii_n} vertices)\n")

    print(f"{'label':<10}{'boundary verts':>16}{'mesh roughness':>18}{'ratio to gii':>14}")
    print("-" * 58)
    for item in args.meshes:
        label, path = item.split("=", 1)
        nodes, tets = read_mesh(path)
        b_verts, b_faces = boundary_surface_from_mesh(nodes, tets)
        mesh_r, mesh_n = roughness(b_verts, b_faces)
        ratio = mesh_r / gii_r if gii_r else float("nan")
        print(f"{label:<10}{mesh_n:>16}{mesh_r:>18.5f}{ratio:>14.3f}")


if __name__ == "__main__":
    main()
