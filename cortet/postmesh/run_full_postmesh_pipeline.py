#!/usr/bin/env python3
r"""run_full_postmesh_pipeline.py -- post-mesh boundary smoothing, end to end.

    [optional] generate the mesh
        -> smooth the boundary  (--method surfacepreserving | wb)
        -> convert to CARP
        -> clean, 2 passes at q > 0.7 and 0.5
        -> convert back to .mesh
        -> report final quality, roughness and boundary displacement

Two rules are enforced rather than documented. Smoothing always re-cleans:
it moves boundary vertices away from the positions the earlier cleaning
optimised around, and an unrecleaned mesh reaches q_max ~ 0.92. And a third
cleaning pass at 0.3 after smoothing is refused: it does not converge and
leaves the mesh worse than before that pass began.

    python3 run_full_postmesh_pipeline.py \
        --mesh subject_raw.mesh --gii input.surf.gii \
        --meshtool ~/meshtool/meshtool --out-dir postmesh/

    python3 run_full_postmesh_pipeline.py \
        --gii input.surf.gii --cell-size 0.4 \
        --meshtool ~/meshtool/meshtool --out-dir postmesh/

--method surfacepreserving (default) uses pymeshlab in-process. --method wb
uses Connectome Workbench and reproduces the published Section 4.4 figures.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_script(name, here_dir=None):
    """Locate a dependent script, whether it sits alongside this one or in a
    sibling folder of the cortet/ layout.

    Search order: same folder, then each sibling one level up. Returns the
    first path that exists; if none do, returns the same-folder guess so the
    caller's subprocess fails loudly rather than this guessing silently.
    """
    base = here_dir if here_dir is not None else HERE
    same_folder = os.path.join(base, name)
    if os.path.exists(same_folder):
        return same_folder

    parent = os.path.join(base, os.pardir)
    for sibling in ("verify", "postmesh", "generation", "surface"):
        candidate = os.path.join(parent, sibling, name)
        if os.path.exists(candidate):
            return candidate

    return same_folder  # not found anywhere searched; let the caller's
                         # subprocess call fail loudly rather than guessing


def run(cmd, label, allow_fail=False):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_fail:
        print(f"\nSTEP FAILED: {label} (exit {result.returncode})", file=sys.stderr)
        print("If this is step 0 (generation) or a meshtool step failing with", file=sys.stderr)
        print("missing-library errors, see ENVIRONMENT.md -- the HPC module", file=sys.stderr)
        print("environment must be sourced in your shell BEFORE running this script.",
              file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode == 0


def validate_args(args):
    """Raises ValueError with a clear message if the argument combination is
    invalid. Separated from main() so this logic can be unit-tested without
    going through sys.exit(). Only --gii is a hard requirement -- --mesh vs
    --pipeline-script is resolved separately by resolve_pipeline_script(),
    since not giving either should fall back to the included generation
    script, not error out."""
    if not args.gii:
        raise ValueError("--gii is required")


def resolve_pipeline_script(args):
    """If neither --mesh nor --pipeline-script was given, fall back to the
    generation script included in this package. Returns the path to use, or
    None if --mesh was given (meaning generation should be skipped entirely).
    Kept separate from main() so this fallback logic is directly testable."""
    if args.mesh:
        return None  # --mesh given: skip generation, don't need a script
    if args.pipeline_script:
        return args.pipeline_script
    return os.path.join(HERE, os.pardir, "generation", "generate_tetrahedral_mesh.py")


def selftest():
    """No real pipeline/meshtool/wb_command needed -- just confirms the
    orchestrator's own argument validation and fallback-resolution logic."""
    ap = build_parser()

    # case 1: missing --gii entirely -- must fail validation
    args_bad = ap.parse_args(["--mesh", "raw.mesh", "--out-dir", "d"])
    try:
        validate_args(args_bad)
        raised = False
    except ValueError:
        raised = True
    assert raised, "should require --gii"

    # case 2: --gii given, --mesh given -- validates fine; resolve_pipeline_script
    # should return None (generation skipped)
    args2 = ap.parse_args(["--mesh", "raw.mesh", "--gii", "x.gii", "--out-dir", "d"])
    validate_args(args2)  # should not raise
    assert resolve_pipeline_script(args2) is None, \
        "giving --mesh should skip generation (resolve to None)"

    # case 3: --gii given, --pipeline-script given explicitly -- should use it verbatim
    args3 = ap.parse_args(["--pipeline-script", "custom_gen.py", "--gii", "x.gii", "--out-dir", "d"])
    validate_args(args3)  # should not raise
    assert resolve_pipeline_script(args3) == "custom_gen.py"

    # case 4: --gii given, NEITHER --mesh nor --pipeline-script -- should fall
    # back to the included generation script (this is the "one hand" default)
    args4 = ap.parse_args(["--gii", "x.gii", "--out-dir", "d"])
    validate_args(args4)  # should not raise -- this is now a VALID combination
    fallback = resolve_pipeline_script(args4)
    expected = os.path.join(HERE, os.pardir, "generation", "generate_tetrahedral_mesh.py")
    assert fallback == expected, f"expected fallback to {expected}, got {fallback}"

    # case 5: find_script() resolves a dependent script in both layouts
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # --- layout A: flat (everything in one folder) ---
        flat_dir = os.path.join(d, "flat_scripts")
        os.makedirs(flat_dir)
        flat_target = os.path.join(flat_dir, "compare_h_direct.py")
        open(flat_target, "w").close()

        found_flat = find_script("compare_h_direct.py", here_dir=flat_dir)
        assert found_flat == flat_target, \
            f"flat layout: expected {flat_target}, got {found_flat}"

        # --- layout B: reorganized (nested, sibling verify/ folder) ---
        nested_root = os.path.join(d, "cortet")
        postmesh_dir = os.path.join(nested_root, "postmesh")
        verify_dir = os.path.join(nested_root, "verify")
        os.makedirs(postmesh_dir)
        os.makedirs(verify_dir)
        nested_target = os.path.join(verify_dir, "compare_h_direct.py")
        open(nested_target, "w").close()

        found_nested = find_script("compare_h_direct.py", here_dir=postmesh_dir)
        assert os.path.realpath(found_nested) == os.path.realpath(nested_target), \
            f"nested layout: expected {nested_target}, got {found_nested}"

        # --- confirm graceful fallback when the script exists nowhere searched ---
        empty_dir = os.path.join(d, "empty_here")
        os.makedirs(empty_dir)
        found_none = find_script("totally_missing_script.py", here_dir=empty_dir)
        assert found_none == os.path.join(empty_dir, "totally_missing_script.py")

    print("selftest PASS: --gii is correctly required; when neither --mesh")
    print("               nor --pipeline-script is given, resolution correctly")
    print("               falls back to the included generation script;")
    print("               explicit --mesh and --pipeline-script are both")
    print("               respected when given.")
    print("               (Cannot test the actual generation/smoothing/")
    print("               cleaning subprocess calls without real pipeline/")
    print("               wb_command/meshtool binaries -- see the other")
    print("               scripts' own --selftest for the logic each step")
    print("               actually depends on.)")
    print("               find_script() verified against BOTH the flat and")
    print("               reorganized (nested sibling-folder) layouts using")
    print("               real temporary directories on disk -- this is the")
    print("               exact check that would have caught the 18/18")
    print("               Step 4a/4b/6 failure before it happened.")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="an already-generated raw mesh (skips step 0)")
    ap.add_argument("--pipeline-script", default=None,
                     help="path to the mesh-generation script; if not given "
                          "and --mesh is also not given, falls back to "
                          "cortet/generation/generate_tetrahedral_mesh.py")
    ap.add_argument("--gii", required=False, help="the input surface")
    ap.add_argument("--cell-size", type=float, default=0.6,
                     help="passed to --pipeline-script as --cell-size, if generating")
    ap.add_argument("--reproject", action="store_true",
                     help="pass --reproject through to --pipeline-script, if generating")
    ap.add_argument("--method", choices=["surfacepreserving", "wb"],
                     default="surfacepreserving",
                     help="which post-mesh boundary smoother to use. "
                          "surfacepreserving (default): pymeshlab's surface-"
                          "preserving Laplacian, applied in-process. "
                          "wb: Workbench wb_command -surface-smoothing "
                          "(needs wb_command installed), the method behind "
                          "the published Section 4.4 figures.")
    ap.add_argument("--wb-command", default="wb_command",
                     help="only used if --method=wb")
    ap.add_argument("--smooth-angle", type=float, default=60.0,
                     help="only used if --method=surfacepreserving; inert "
                          "above ~20 degrees on cortical surfaces, leave at 60")
    ap.add_argument("--meshtool", default="meshtool")
    ap.add_argument("--out-dir", default="postmesh_output")
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--interior-shells", type=int, default=2)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.5],
                     help="clean thresholds; default 0.7,0.5 -- do not add 0.3")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    try:
        validate_args(args)
    except ValueError as e:
        ap.error(str(e))

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- STEP 0: generate the raw mesh, if requested (or by default) ----
    if args.mesh:
        raw_mesh = args.mesh
        print(f"Using existing raw mesh: {raw_mesh} (skipping generation)")
    else:
        pipeline_script = resolve_pipeline_script(args)
        if not os.path.exists(pipeline_script):
            print(f"ERROR: generation script not found: {pipeline_script}", file=sys.stderr)
            sys.exit(1)
        print(f"Generating from: {pipeline_script}")

        gii_base = os.path.splitext(os.path.splitext(os.path.basename(args.gii))[0])[0]
        raw_mesh = os.path.join(args.out_dir, f"{gii_base}_raw.mesh")
        work_dir = os.path.join(args.out_dir, f"_work_{gii_base}")
        os.makedirs(work_dir, exist_ok=True)

        gen_cmd = [sys.executable, pipeline_script,
                   "-i", args.gii, "-o", raw_mesh,
                   "--cell-size", str(args.cell_size),
                   "--meshtool-path", args.meshtool,
                   "--work-dir", work_dir]
        if args.reproject:
            gen_cmd.append("--reproject")
        run(gen_cmd, "STEP 0/6: mesh generation")

    base = os.path.splitext(os.path.basename(raw_mesh))[0]
    # name the intermediate after the method that produced it
    method_tag = {"wb": "wbsmoothed", "surfacepreserving": "spsmoothed"}[args.method]
    smoothed_mesh = os.path.join(args.out_dir, f"{base}_{method_tag}.mesh")
    carp_base = os.path.join(args.out_dir, f"{base}_carp")
    final_mesh = os.path.join(args.out_dir, f"{base}_FINAL.mesh")

    # ---- STEP 1: smooth (method selectable via --method) ----
    if args.method == "surfacepreserving":
        run([sys.executable, find_script("smooth_mesh_boundary_surfacepreserving.py"),
             "--mesh", raw_mesh, "--out", smoothed_mesh,
             "--iterations", str(args.iterations), "--angle", str(args.smooth_angle),
             "--interior-shells", str(args.interior_shells)],
            "STEP 1/6: boundary smoothing (surface-preserving pymeshlab filter)")
    else:  # wb
        run([sys.executable, find_script("smooth_mesh_boundary_wb.py"),
             "--mesh", raw_mesh, "--out", smoothed_mesh,
             "--wb-command", args.wb_command,
             "--strength", str(args.strength), "--iterations", str(args.iterations),
             "--interior-shells", str(args.interior_shells)],
            "STEP 1/6: boundary smoothing (Workbench wb_command)")

    # ---- STEP 2+3: convert to CARP, clean, convert back ----
    thr_args = [str(t) for t in args.thresholds]
    run([sys.executable, find_script("mesh_to_carp_and_clean.py"),
         "--mesh", smoothed_mesh, "--carp-base", carp_base,
         "--clean", "--meshtool", args.meshtool,
         "--thresholds", *thr_args,
         "--final-mesh", final_mesh],
        "STEP 2-3/6: CARP conversion + meshtool cleaning "
        f"(thresholds {args.thresholds})")

    # ---- STEP 4: verify against the input .gii ----
    run([sys.executable, find_script("compare_h_direct.py"),
         "--gii", args.gii, "--meshes", f"raw={raw_mesh}", f"final={final_mesh}"],
        "STEP 4a/6: roughness check (raw AND final mesh vs input .gii)")

    run([sys.executable, find_script("check_boundary_displacement.py"),
         "--gii", args.gii, "--mesh", final_mesh,
         "--out", os.path.join(args.out_dir, f"{base}_displacement.csv")],
        "STEP 4b/6: boundary displacement check (final mesh vs input .gii)")

    # ---- STEP 5: finalize into the COMPLETE, real BrainGrowth format ----
    # (boundary-face section + surface-first reordering,
    # "the .mesh format has a third section" for why this step exists)
    braingrowth_ready_mesh = os.path.join(args.out_dir, f"{base}_BRAINGROWTH_READY.mesh")
    run([sys.executable, find_script("finalize_for_braingrowth.py"),
         "--mesh", final_mesh, "--out", braingrowth_ready_mesh],
        "STEP 5/6: finalize into the complete BrainGrowth format "
        "(boundary faces + surface-first reordering)")

    # ---- STEP 6: verify the finalized file actually IS surface-first ----
    # (a self-check on Step 5's own output, not a leap of faith -- confirms
    # the property the solver's tetra_normals() depends on actually holds
    # in the file you're about to simulate with, not just that finalize()
    # ran without error)
    run([sys.executable, find_script("check_node_reordering.py"),
         "--mesh", braingrowth_ready_mesh],
        "STEP 6/6: verify surface-first node ordering in the final file")

    print(f"\n{'='*70}")
    print(f"DONE. Raw mesh:              {raw_mesh}")
    print(f"      Final (working format): {final_mesh}")
    print(f"      BRAINGROWTH-READY:      {braingrowth_ready_mesh}")
    print("^ This last file is the one to actually use in simulation -- it is")
    print("  the only one with the complete format (boundary faces + surface-")
    print("  first node order) that the solver's loader expects, and Step 6")
    print("  above already confirmed the ordering property holds for THIS file.")
    print("Review the roughness and displacement numbers printed above against")
    print("your own thresholds before using this mesh in simulation.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

