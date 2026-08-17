# Environment requirements

Both mesh generation and meshtool's cleaning step depend on a specific
module environment. Missing this is the single most common cause of a
step failing (or, worse, succeeding but silently doing nothing / producing
subtly wrong output) throughout this project.

## Confirmed exact requirements (from the generation script's own docstring)

These are the literal requirements stated in
`generate_tetrahedral_mesh.py` itself (confirmed by reading the
file, not paraphrased):

```bash
module load eigen/3.4.0-gcc-13.2.0
module load gmp/6.2.1-gcc-13.2.0
module load boost/1.83.0-gcc-13.2.0
# Do NOT load the cgal module -- pygalmesh uses local CGAL 5.6.1 headers
```

Plus:
```bash
pip install pygalmesh nibabel trimesh meshio gmsh --user --no-deps
```

And:
- `meshtool` binary expected at `~/meshtool/meshtool` by default (override
  with `--meshtool-path`)
- CGAL 5.6.1 headers at `~/.local/include/CGAL/` (header-only)
- `pygalmesh` rebuilt from source against CGAL 5.6.1 with
  `--no-build-isolation`
- Gmsh's wheel may need Mesa/GLU on `LD_LIBRARY_PATH`. If `import gmsh` fails
  with a missing `libGLU`/`libGL`, set `CORTET_GL_LIBS` to a colon-separated
  list of the directories holding them and the pipeline prepends it. Not needed
  on a desktop install.

## The full module-loading block used throughout this project

```bash
module load anaconda3/2022.10-gcc-13.2.0
module load eigen/3.4.0-gcc-13.2.0 gmp/6.2.1-gcc-13.2.0 boost/1.83.0-gcc-13.2.0
export CORTET_GL_LIBS=<mesa-glu>/lib:<mesa>/lib      # only if gmsh needs it
```

Source this in your shell BEFORE running `run_full_postmesh_pipeline.py`
(or any individual script in this package that calls your generation
pipeline or `meshtool`). None of these scripts load modules for you.

The exact module names/versions above are specific to the HPC system used
in this project (King's CREATE) and will very likely differ on your system
-- what matters is confirming, once, which modules your `meshtool` and your
generation pipeline actually need, then sourcing that block every time.

## How to (re-)discover this if you're not sure

1. Get a bare `meshtool query quality` call working interactively first,
   note exactly what's loaded (`module list`) in that working shell.
2. Test the same command in a fresh, non-interactive shell (or inside an
   `sbatch` script) with NO modules loaded. If it fails or behaves
   differently, that confirms the environment -- not the command -- is what
   matters.
3. Load the modules from step 1 into that fresh shell one at a time until
   it matches the interactive result.
4. Use that exact block everywhere.

## Two failure modes specific to this project, worth knowing about

**meshtool's `-odat` per-element export is especially sensitive to this** --
it can exit with status 0 and produce empty/garbage output outside the
correct environment, with no error message pointing at the real cause. Not
used directly by the scripts in THIS package, but relevant if you extend
this pipeline to capture per-element quality data.

**Parallel jobs sharing a work directory can corrupt each other**,
independent of the module environment -- this is a different failure mode
that produced a mesh with genuinely wrong quality
(qmax=0.736 instead of the expected ~0.49) purely from a work-directory
collision with another job running at the same time. Always give each
generation run (and, if running multiple subjects in parallel, each
subject) its own isolated work directory.

## Do not run generation and any other job against the same work directory concurrently

`run_full_postmesh_pipeline.py`'s generation step creates its own
`_work_<name>` subdirectory under `--out-dir` for exactly this reason.
Don't override this to point multiple concurrent runs at the same path.

## Before you submit a job: four things that fail instantly if you skip them

Each of these cost real time on this project. They are cheap to check and each
produces a failure whose error message does not name the actual cause.

**1. Create the working directory yourself.** The generation script does not
create its own `--work-dir`; it fails immediately if the path does not exist.
The numbered stages in `run/` do create it for you, but a bare call to
`cortet/generation/generate_tetrahedral_mesh.py` does not.

```bash
mkdir -p /scratch/users/$USER/<run name>
```

**2. Create the SLURM log directory before `sbatch`, not after.** Slurm opens
the log file when the job starts, so a missing directory makes every task in an
array die instantly with nothing useful written anywhere — the logs that would
have told you why are the files it could not create.

```bash
mkdir -p ~/<run name>/logs && sbatch <script>.sh
```

**3. Work on scratch, not in your home directory.** On CREATE, `$HOME` has a
50 GB quota and `/scratch` has 200 GB. Mesh generation at h = 0.4 writes
multi-gigabyte intermediates, and exhausting the home quota mid-run fails in
places that look nothing like a disk problem — a `csv.writer` call raising
`OSError: Disk quota exceeded` part-way through a results file, for instance.
Point `--work-dir` and `--out-dir` at scratch from the start.

**4. Give every concurrent run its own work directory.** See the section above:
a shared work directory between parallel jobs silently produced a mesh with the
wrong quality.

And once the environment is set up, confirm it rather than assuming it:

```bash
python3 run/00_check_environment.py --meshtool ~/meshtool/meshtool
```

That measures a single regular tetrahedron, whose quality must come back near 0.
`meshtool` can exit 0 and emit wrong output outside the correct module
environment, so checking the binary exists is not enough.

## Reproducing on a workstation, and how closely it matches

The whole pipeline builds and runs off-cluster. On Ubuntu 22.04 with CGAL 5.6.1 headers in
`~/.local/include`, the missing pieces were:

```bash
# gmp and mpfr runtime libs exist, but the linker names (libgmp.so, libmpfr.so)
# come from -dev packages. Without root, symlink them somewhere you own:
mkdir -p ~/.local/lib
ln -sf /usr/lib/x86_64-linux-gnu/libgmp.so.10  ~/.local/lib/libgmp.so
ln -sf /usr/lib/x86_64-linux-gnu/libmpfr.so.6  ~/.local/lib/libmpfr.so

export CPATH="$HOME/.local/include:$CPATH"          # CGAL + Eigen headers
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH" # link time
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"  # run time
pip install --user pygalmesh

# meshtool is not pip-installable; build it
git clone https://bitbucket.org/aneic/meshtool.git && cd meshtool && make
```

Then `python3 run/00_check_environment.py --meshtool <path>` should report a clean environment.

### How closely a workstation run reproduces the cluster

Measured on `subject_GA21.86_left` at `h = 0.6` from the same `sp30` surface, against the cluster numbers
in `results/table2_full_matrix.csv`:

| | cluster | workstation |
|---|---|---|
| nodes | 59,174 | **59,174** — exact |
| tets | 326,083 | 326,101 (+18, +0.0055%) |
| `q̄` full pipeline | 0.1289 | 0.1310 |
| `q_max` full pipeline | 0.4621 | 0.4809 — passes, as there |

**The node count matches exactly and the element count does not.** That combination is informative:
CGAL placed the same vertices and connected eighteen of them differently, which is Delaunay
tie-breaking among near-cospherical points — precisely where compiler flags and library builds show
through. It is a build difference, not a code difference, and it enters at **S1**: the ablation shows
node and element counts identical across CGAL only, +Gmsh and Full, so nothing downstream can change
them.

**Do not expect bit-exact agreement across different builds of CGAL, pygalmesh, Gmsh and meshtool.**
Expect the same conclusions. The ablation reproduced all three published `q_max` ranges
(CGAL only 0.767 in 0.746–0.820; +Gmsh 0.720 in 0.685–0.802; Full 0.481 in 0.462–0.561) and the same
categorical result: 280 solver-breaking elements before S4, zero after.

### One place the difference does bite

Post-mesh smoothing followed by the two-pass clean:

| | cluster | workstation |
|---|---|---|
| after smoothing, before cleaning | ~0.921 | 0.8795 (194 elements above threshold) |
| after the two-pass clean | 0.4981, 0 above | **0.6547, 2 above** |

Smoothing behaves the same on both — badly, by design, which is why the re-clean is mandatory. The
**clean** is what differs: this workstation's `meshtool` (built from HEAD) leaves two of 326,101
elements above the threshold where the cluster's older build removes them. `meshtool clean quality`
is an iterative heuristic, so version differences show up exactly here.

`run/05` correctly **failed** that mesh rather than passing it on. If you see this, either re-run the
clean with an additional pass at 0.5, or use the cluster build — do not simulate with it.

### The post-mesh clean is meshtool-build sensitive — isolate it before blaming the smoother

Post-mesh smoothing followed by the two-pass clean did not recover on a workstation build. The
failure was isolated by elimination, and the result is worth recording because the obvious suspect is
the wrong one.

All three smoothers, same input mesh, same subject at `h = 0.6`:

| `--method` | `q_max` after the two-pass clean | |
|---|---|---|
| `surfacepreserving` | 0.6547 | FAIL |
| `wb` | 0.6693 | FAIL |
| cluster, `surfacepreserving` | **0.4981** | PASS |

**Three different smoothers, one shared failure — so it is not the smoother.** Smoothing itself
behaved correctly on all three: `q_max` went to 0.88 before cleaning, the expected "broken until
re-cleaned" state.

Further elimination:

- **Extra cleaning passes do not help.** `--thresholds 0.7 0.5 0.5 0.5` gave 0.6649, no better than
  two passes. The clean converges somewhere different, it does not merely stop early.
- **The code is identical.** The postmesh scripts were diffed against the exact package the cluster
  ran; every difference is a comment. Parameters match too — the matrix run passes no `--method`,
  `--iterations` or `--thresholds`, so both used the same defaults.
- **The Python environment was matched** — numpy 1.24.4, scipy 1.9.1, trimesh 4.11.2, meshio 5.3.5,
  nibabel 5.3.3, pygalmesh 0.10.7 — and made no difference.

What is left is the `meshtool` binary itself. `meshtool clean quality` is an iterative heuristic, and
the workstation build (bitbucket HEAD, commit `394e8c7`, 25 May 2026) converges differently from the
cluster's older build.

**If your post-mesh clean will not recover a mesh, check your `meshtool` build before changing
anything else.** `run/05` fails such a mesh rather than passing it on, which is the intended
behaviour — do not work around it by loosening the threshold.

To pin this down properly, record the commit your `meshtool` was built from:

```bash
cd ~/meshtool && git log -1 --format="%h %ad" --date=short
```

That is the one version this project has never captured, and it is the one that matters here.
