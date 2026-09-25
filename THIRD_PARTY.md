# Third-party tools

CORTET orchestrates external tools; it does not include or redistribute them.
The MIT licence in `LICENSE` covers this repository's own source code only. Each tool
keeps its own licence, and you obtain it from its own distributor:

| Tool | Licence | Where |
|---|---|---|
| CGAL (via pygalmesh) | GPLv3+ / commercial | <https://www.cgal.org> |
| Gmsh | GPLv2+ | <https://gmsh.info> |
| meshtool | Apache-2.0 | <https://github.com/aureliobarbosa/meshtool> |
| TetGen | AGPLv3 / commercial | <https://wias-berlin.de/software/tetgen> |
| Connectome Workbench | GPLv2 | <https://humanconnectome.org> |
| PyMeshLab | GPLv3 | <https://pymeshlab.readthedocs.io> |
| meshio, NiBabel | MIT | |
| Trimesh | MIT | |

Note that a pipeline you assemble at run time from these tools may, as a
combined work, be subject to their terms. If you plan to redistribute a bundle
that includes them, check the licence of each tool you bundle.
