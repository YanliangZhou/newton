# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Training scenario v1.4 (multi-world cloth + manipulator batch)
#
# Evolved from ``example_cloth_franka.py``: ``N`` independent worlds (default 100) on a
# configurable grid—each with one arm, one table, one procedural cloth panel—and shared
# batched Cartesian control. File name ``example_training_v1_4.py`` (CLI short name
# ``training_v1_4``): Python module names cannot contain ``.``, so ``v1.4`` is encoded as
# ``v1_4`` in the filename.
#
# Main features:
#   - One articulation per world (URDF from :func:`demo_arm_pool`)
#   - One table and one cloth panel per world (randomized shape / stiffness / density)
#   - Batched end-effector targets with per-world Jacobian solve
#
# Notes:
#   - Simulation still uses centimeter scale, like the original example.
#   - By default, the robots share the same nominal motion pattern, offset into each cell.
#   - Jacobian solve is per-world (see ``compute_body_jacobian_for_world`` +
#     ``generate_control_joint_qd``), so each world uses its own current state.
#   - Arm/animation *pools* in this file mainly affect metadata and which URDF is loaded;
#     the keyframe trajectory driving ``generate_control_joint_qd`` is still a single shared
#     table until you branch on ``animation_id`` (or similar) in code.
#
# Demo bundle (single-file API for early experiments; see ``__all__`` and
# :class:`TrainingDemoV1_4Api` for a stable import surface when vendoring).
# Human-oriented usage (CLI, custom URDF / animation pools): ``TRAINING_V1_4_README.md``
# in this directory.
#   - Edit :data:`DEMO_SCENARIO` or pass ``--demo-world-count`` / ``--demo-grid-*``;
#     use :func:`make_demo_scenario` when constructing from code.
#   - ``--demo-cloth-grid-density S`` scales random panel subdivisions (``nx``, ``ny``);
#     still independent random counts per axis; trapezoid/skew use the same ``(u,v)`` grid.
#   - ``--demo-cloth-panel-rng-entropy`` draws each world's cloth panel RNG seed from OS
#     entropy at startup (different procedural cloth each run; not bit-reproducible).
#   - ``--demo-no-display`` turns off interactive viewers (GL / Viser / Rerun);
#     use ``--viewer usd --output-path …`` to record each frame to USD (default).
#     Use ``--no-demo-write-usd`` to keep ``ViewerUSD`` (frame count / FPS) but skip
#     per-frame ``log_state`` / ``log_shapes`` I/O for benchmarking.
#     Per-world JSON is written at shutdown by default; skip with ``--no-demo-write-json``.
#     For training: USD + JSON with no window, use ``usd`` + ``--demo-no-display`` (not ``null``).
#   - Edit :func:`demo_arm_pool` / :func:`demo_animation_pool` for asset lists;
#     :func:`assign_demo_world_assets` samples with replacement when pools are
#     shorter than ``world_count``.
#   - Dynamic trajectories and meshes go to USD (``--output-path`` with ``--viewer usd``);
#     per-world sidecar JSON files share the same directory as the metadata **base** path
#     from ``--demo-metadata-json`` (stem ``*_world_WWWW``; see :func:`write_demo_metadata`).
#   - **Session folders (recommended for sweeps / many processes):** ``--demo-output-root DIR``
#     assigns ``--output-path`` and ``--demo-metadata-json`` under one new subdirectory
#     ``DIR/run_<YYYYMMDD>_<HHMMSS>_<pid>/`` (:func:`resolve_demo_recording_session_dir`).
#     That folder always contains ``recording.usd`` plus ``example_training_v1_4_meta.json``
#     (fixed names); per-world JSONs are ``example_training_v1_4_meta_world_0000.json``, …
#     next to the base file. Timestamp + process id avoids collisions when several jobs
#     start together; point each worker at the same ``DIR`` and every process writes its own
#     run directory (see :func:`apply_demo_recording_session_paths`).
#
# USD vs JSON (reproducibility):
#   - ``ViewerUSD`` time-samples whatever :meth:`newton.viewer.ViewerBase.log_state`
#     records for the logged :class:`~newton.State` (particle positions, body
#     transforms, etc.) plus this example's ``log_shapes`` for tables. See
#     ``newton/_src/viewer/viewer_usd.py`` and the base ``log_state`` implementation.
#   - Each ``*_world_WWWW.json`` holds everything else needed to rebuild the same
#     procedural cloth and layout: RNG seeds, sampled cloth parameters, asset ids,
#     solver settings, and the embedded robot keyframe table. Together with the USD
#     timeline this is intended to reproduce the demo setup; bit-identical physics
#     on GPU may still depend on device and Warp version.
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.viewer import ViewerFile, ViewerUSD
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD


# ---------------------------------------------------------------------------
# Demo scenario + recording (all configuration lives in this file)
# ---------------------------------------------------------------------------
#
# Stable imports for downstream packages: see ``__all__`` and :class:`TrainingDemoV1_4Api`.


@dataclass
class DemoScenarioConfig:
    world_count: int = 100
    """Number of parallel worlds (cloth + table + arm + one animation each)."""
    grid_rows: int = 10
    grid_cols: int = 10
    cell_spacing_x: float = 180.0
    """Cell spacing along X in simulation units (cm in this example)."""
    cell_spacing_y: float = 180.0
    """Cell spacing along Y in simulation units (cm in this example)."""
    asset_assignment_seed: int | None = 12345
    """Seed for assigning arm/animation assets to worlds; ``None`` is non-reproducible."""
    enable_visual_display: bool = True
    """If ``False``, skip on-screen / interactive viewers only; file-backed recorders still log."""
    write_usd: bool = True
    """If ``False`` and viewer is :class:`~newton.viewer.ViewerUSD`, skip mesh/state USD export (perf test)."""
    write_metadata_json: bool = True
    """If ``False``, skip per-world JSON at shutdown."""
    cloth_mesh_density_scale: float = 1.0
    """Scale ``nx`` / ``ny`` sampling bounds (``<1`` coarser, ``>1`` finer). Random ``nx`` ≠ ``ny``."""
    cloth_grid_nx_min: int = 14
    cloth_grid_nx_max: int = 46
    """Inclusive bounds for random ``nx`` before :attr:`cloth_mesh_density_scale` is applied."""
    cloth_grid_ny_min: int = 14
    cloth_grid_ny_max: int = 46
    """Inclusive bounds for random ``ny`` before :attr:`cloth_mesh_density_scale` is applied."""
    cloth_panel_rng_use_entropy: bool = False
    """If ``True``, each world's cloth panel NumPy RNG seed is sampled from OS entropy at
    :class:`Example` construction (different procedural cloth each process start; not
    reproducible from seed alone). If ``False``, seeds are ``1000 + world_index`` (stable
    across runs for the same ``world_count``).
    """
    cloth_panel_fixed_grid_nx: int | None = None
    """If set together with :attr:`cloth_panel_fixed_grid_ny`, every panel uses the **same**
    vertex counts ``nx``×``ny`` (independent of each panel's random width/height). This does
    **not** keep physical cell size uniform across panels. For that, use
    :attr:`cloth_panel_target_cell_cm` instead. Base counts are multiplied by
    :attr:`cloth_mesh_density_scale` (rounded, clamped to at least 4 per axis).
    """
    cloth_panel_fixed_grid_ny: int | None = None
    """Paired with :attr:`cloth_panel_fixed_grid_nx`; see that field."""
    cloth_panel_target_cell_cm: float | None = None
    """If set, ``nx`` and ``ny`` are chosen from each panel's random ``width`` / ``height`` so
    mean edge spacing along the panel axes is approximately this value (centimeters in this
    example), then clamped to at most the scaled ``cloth_grid_*_max`` caps. Mutually exclusive
    with :attr:`cloth_panel_fixed_grid_nx` / :attr:`cloth_panel_fixed_grid_ny`. Finer cells when
    :attr:`cloth_mesh_density_scale` is larger (target is divided by the scale).
    """


@dataclass
class DemoRecordingPaths:
    usd_path: str | None = None
    """USD output; typically from ``--output-path`` when using the USD viewer."""
    metadata_json_path: str = "example_training_v1_4_meta.json"
    """Base path for the aggregate JSON stem; per-world files are ``{stem}_world_{i:04d}{suffix}`` in the same directory."""

    def resolved_metadata_path(self) -> Path:
        return Path(self.metadata_json_path).expanduser().resolve()

    def per_world_metadata_path(self, world_index: int) -> Path:
        """Path for one world's sidecar JSON next to :meth:`resolved_metadata_path`."""
        base = self.resolved_metadata_path()
        stem, suf = base.stem, base.suffix
        if not suf:
            suf = ".json"
        return base.with_name(f"{stem}_world_{world_index:04d}{suf}")


# Default filenames inside a per-run session directory (USD + JSON share one folder / run id).
DEMO_SESSION_DIR_USD_NAME = "recording.usd"
DEMO_SESSION_DIR_METADATA_BASE = "example_training_v1_4_meta.json"


def resolve_demo_recording_session_dir(output_root: str | os.PathLike[str]) -> Path:
    """Create and return ``<output_root>/run_<YYYYMMDD>_<HHMMSS>_<pid>/``.

    Each run gets its own subdirectory under ``output_root`` so USD, the metadata base JSON,
    and all ``*_world_*.json`` sidecars stay in one place. The ``pid`` suffix keeps concurrent
    processes from clashing when they start in the same wall-clock second.
    """
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out = root / f"run_{stamp}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_demo_recording_session_paths(args: argparse.Namespace) -> None:
    """If ``--demo-output-root`` is set, redirect USD + metadata into a new session directory.

    Sets ``args.output_path`` to ``<session>/recording.usd`` and ``args.demo_metadata_json`` to
    ``<session>/example_training_v1_4_meta.json`` with ``session = resolve_demo_recording_session_dir(root)``,
    and sets ``args.demo_output_session_dir`` to ``session`` for logging. No-op if the flag is unset.

    Use one shared ``--demo-output-root`` across many processes so each run auto-numbers into its
    own folder; per-world JSON paths follow the metadata basename inside that folder.
    """
    root = getattr(args, "demo_output_root", None)
    if not root:
        return
    session_dir = resolve_demo_recording_session_dir(output_root=root)
    args.output_path = str(session_dir / DEMO_SESSION_DIR_USD_NAME)
    args.demo_metadata_json = str(session_dir / DEMO_SESSION_DIR_METADATA_BASE)
    setattr(args, "demo_output_session_dir", str(session_dir))


def _argv_with_demo_session_paths(
    argv: list[str],
    *,
    output_path: str,
    metadata_json: str,
) -> list[str]:
    """Strip prior ``--output-path`` / ``--demo-metadata-json`` and append resolved values (``init`` re-parses)."""
    out: list[str] = []
    i = 0
    prog = argv[0] if argv else ""
    if prog:
        out.append(prog)
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--output-path":
            i += 2
            continue
        if a.startswith("--output-path="):
            i += 1
            continue
        if a == "--demo-metadata-json":
            i += 2
            continue
        if a.startswith("--demo-metadata-json="):
            i += 1
            continue
        out.append(a)
        i += 1
    out.extend(["--output-path", output_path, "--demo-metadata-json", metadata_json])
    return out


@dataclass
class DemoAssetSpec:
    id: str
    """Stable string id stored in metadata."""
    source: str
    """URDF filesystem path, or ``builtin:franka`` for the bundled Franka download."""


@dataclass
class DemoWorldAssetAssignment:
    world_index: int
    arm_id: str
    arm_source: str
    animation_id: str
    animation_source: str
    arm_pool_index: int
    """Index into the arm pool for this draw (reproducibility)."""
    animation_pool_index: int
    """Index into the animation pool for this draw."""


DEMO_SCENARIO = DemoScenarioConfig()
"""Module-level defaults; adjust here or replace when constructing :class:`Example`."""


def demo_scaled_cloth_grid_bounds(cfg: DemoScenarioConfig) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``((nx_lo, nx_hi), (ny_lo, ny_hi))`` used to sample panel subdivisions.

    Bounds from :attr:`DemoScenarioConfig.cloth_grid_nx_min` / ``*_max`` are multiplied by
    :attr:`DemoScenarioConfig.cloth_mesh_density_scale`, clamped to at least 4 vertices per axis.
    """
    s = float(cfg.cloth_mesh_density_scale)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("cloth_mesh_density_scale must be a positive finite float")

    def one_axis(lo: int, hi: int) -> tuple[int, int]:
        lo_r = max(4, min(lo, hi))
        hi_r = max(lo_r, max(lo, hi))
        lo_s = max(4, int(round(lo_r * s)))
        hi_s = max(lo_s, int(round(hi_r * s)))
        return lo_s, hi_s

    return one_axis(cfg.cloth_grid_nx_min, cfg.cloth_grid_nx_max), one_axis(
        cfg.cloth_grid_ny_min, cfg.cloth_grid_ny_max
    )


def demo_viewer_writes_frame_log(viewer) -> bool:
    """True for backends that persist per-frame data via :meth:`Example.render` (USD, recording file)."""
    return isinstance(viewer, (ViewerUSD, ViewerFile))


def demo_should_run_render(viewer, enable_visual_display: bool) -> bool:
    """Whether to run the render path: always for USD / ViewerFile; else only if visuals are enabled."""
    if viewer is None:
        return False
    if demo_viewer_writes_frame_log(viewer):
        return True
    return enable_visual_display


def default_demo_grid_for_world_count(world_count: int) -> tuple[int, int]:
    """Pick ``(grid_rows, grid_cols)`` with ``rows * cols == world_count`` (factor near square)."""
    if world_count < 1:
        raise ValueError("world_count must be >= 1")
    r = int(math.isqrt(world_count))
    while r > 0 and world_count % r != 0:
        r -= 1
    if r <= 0:
        return world_count, 1
    return r, world_count // r


def _validate_fixed_cloth_panel_grid(nx: int | None, ny: int | None) -> None:
    if (nx is None) ^ (ny is None):
        raise ValueError("Set both cloth_panel_fixed_grid_nx and cloth_panel_fixed_grid_ny, or neither.")
    if nx is None:
        return
    if nx < 4 or ny < 4:
        raise ValueError("cloth_panel_fixed_grid_nx and cloth_panel_fixed_grid_ny must be >= 4.")


def _validate_demo_cloth_panel_resolution(cfg: DemoScenarioConfig) -> None:
    """Reject incompatible cloth panel resolution settings."""
    _validate_fixed_cloth_panel_grid(cfg.cloth_panel_fixed_grid_nx, cfg.cloth_panel_fixed_grid_ny)
    if cfg.cloth_panel_target_cell_cm is not None:
        if cfg.cloth_panel_fixed_grid_nx is not None:
            raise ValueError(
                "Do not set cloth_panel_target_cell_cm together with cloth_panel_fixed_grid_nx/ny."
            )
        x = float(cfg.cloth_panel_target_cell_cm)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("cloth_panel_target_cell_cm must be a positive finite float when set.")


def make_demo_scenario(
    world_count: int,
    *,
    base: DemoScenarioConfig | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    enable_visual_display: bool | None = None,
    write_usd: bool | None = None,
    write_metadata_json: bool | None = None,
    cloth_mesh_density_scale: float | None = None,
    cloth_grid_nx_min: int | None = None,
    cloth_grid_nx_max: int | None = None,
    cloth_grid_ny_min: int | None = None,
    cloth_grid_ny_max: int | None = None,
    cloth_panel_rng_use_entropy: bool | None = None,
    cloth_panel_fixed_grid_nx: int | None = None,
    cloth_panel_fixed_grid_ny: int | None = None,
    cloth_panel_target_cell_cm: float | None = None,
) -> DemoScenarioConfig:
    """Build a :class:`DemoScenarioConfig` for an arbitrary world count (programmatic API).

    If ``grid_rows`` / ``grid_cols`` are omitted, a factorization of ``world_count`` is chosen
    automatically (same logic as CLI when only ``--demo-world-count`` is set).

    Args:
        write_usd: If set, override :attr:`DemoScenarioConfig.write_usd`.
        write_metadata_json: If set, override :attr:`DemoScenarioConfig.write_metadata_json`.
        cloth_mesh_density_scale: If set, override mesh density (scales ``nx``/``ny`` bounds).
        cloth_grid_nx_min: Optional override for random ``nx`` lower bound.
        cloth_grid_nx_max: Optional override for random ``nx`` upper bound.
        cloth_grid_ny_min: Optional override for random ``ny`` lower bound.
        cloth_grid_ny_max: Optional override for random ``ny`` upper bound.
        cloth_panel_rng_use_entropy: If set, override :attr:`DemoScenarioConfig.cloth_panel_rng_use_entropy`.
        cloth_panel_fixed_grid_nx: If set, override :attr:`DemoScenarioConfig.cloth_panel_fixed_grid_nx`
            (must set ``cloth_panel_fixed_grid_ny`` too).
        cloth_panel_fixed_grid_ny: If set, override :attr:`DemoScenarioConfig.cloth_panel_fixed_grid_ny`.
        cloth_panel_target_cell_cm: If set, override :attr:`DemoScenarioConfig.cloth_panel_target_cell_cm`.
    """
    b = base or DEMO_SCENARIO
    _validate_fixed_cloth_panel_grid(cloth_panel_fixed_grid_nx, cloth_panel_fixed_grid_ny)
    if (grid_rows is None) ^ (grid_cols is None):
        raise ValueError("Set both grid_rows and grid_cols, or neither.")
    if grid_rows is None:
        r, c = default_demo_grid_for_world_count(world_count)
    else:
        r, c = grid_rows, grid_cols
    if r * c != world_count:
        raise ValueError(f"grid_rows * grid_cols ({r * c}) must equal world_count ({world_count}).")
    cfg = replace(b, world_count=world_count, grid_rows=r, grid_cols=c)
    if enable_visual_display is not None:
        cfg = replace(cfg, enable_visual_display=enable_visual_display)
    if write_usd is not None:
        cfg = replace(cfg, write_usd=write_usd)
    if write_metadata_json is not None:
        cfg = replace(cfg, write_metadata_json=write_metadata_json)
    if cloth_mesh_density_scale is not None:
        cfg = replace(cfg, cloth_mesh_density_scale=float(cloth_mesh_density_scale))
    if cloth_grid_nx_min is not None:
        cfg = replace(cfg, cloth_grid_nx_min=cloth_grid_nx_min)
    if cloth_grid_nx_max is not None:
        cfg = replace(cfg, cloth_grid_nx_max=cloth_grid_nx_max)
    if cloth_grid_ny_min is not None:
        cfg = replace(cfg, cloth_grid_ny_min=cloth_grid_ny_min)
    if cloth_grid_ny_max is not None:
        cfg = replace(cfg, cloth_grid_ny_max=cloth_grid_ny_max)
    if cloth_panel_rng_use_entropy is not None:
        cfg = replace(cfg, cloth_panel_rng_use_entropy=bool(cloth_panel_rng_use_entropy))
    if cloth_panel_fixed_grid_nx is not None:
        cfg = replace(
            cfg,
            cloth_panel_fixed_grid_nx=cloth_panel_fixed_grid_nx,
            cloth_panel_fixed_grid_ny=cloth_panel_fixed_grid_ny,
        )
    if cloth_panel_target_cell_cm is not None:
        x = float(cloth_panel_target_cell_cm)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("cloth_panel_target_cell_cm must be a positive finite float when set.")
        cfg = replace(cfg, cloth_panel_target_cell_cm=x)
    _validate_demo_cloth_panel_resolution(cfg)
    return cfg


def add_demo_scenario_args(parser) -> None:
    """CLI overrides for :class:`DemoScenarioConfig` (call before :func:`newton.examples.init`)."""
    parser.add_argument(
        "--demo-world-count",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worlds (default: DEMO_SCENARIO.world_count).",
    )
    parser.add_argument(
        "--demo-grid-rows",
        type=int,
        default=None,
        metavar="R",
        help="Layout rows; must be given with --demo-grid-cols and R*C == world count.",
    )
    parser.add_argument(
        "--demo-grid-cols",
        type=int,
        default=None,
        metavar="C",
        help="Layout cols; must be given with --demo-grid-rows.",
    )
    parser.add_argument(
        "--demo-no-display",
        action="store_true",
        help="Disable GL/Viser/Rerun drawing only; USD (--viewer usd) and ViewerFile still log frames.",
    )
    parser.add_argument(
        "--demo-cloth-grid-density",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Scale cloth panel subdivisions: multiplies random nx/ny bounds from cloth_grid_n*, "
            "divides --demo-cloth-panel-target-cell-cm target edge length, or multiplies fixed "
            "--demo-cloth-panel-fixed-nx/ny base counts when those are set."
        ),
    )
    parser.add_argument(
        "--demo-cloth-panel-target-cell-cm",
        type=float,
        default=None,
        metavar="H",
        help=(
            "Pick nx/ny from each panel's random width/height so mean quad edge length ~H cm along "
            "each axis (same H all worlds; nx/ny vary with panel size). Capped by scaled "
            "cloth_grid_*_max. Incompatible with --demo-cloth-panel-fixed-nx/ny."
        ),
    )
    parser.add_argument(
        "--demo-cloth-panel-fixed-nx",
        type=int,
        default=None,
        metavar="NX",
        help=(
            "Force the same vertex nx on every panel (pair with --demo-cloth-panel-fixed-ny); "
            "cell size in cm still varies with random panel width/height. For ~uniform cell size, "
            "use --demo-cloth-panel-target-cell-cm instead."
        ),
    )
    parser.add_argument(
        "--demo-cloth-panel-fixed-ny",
        type=int,
        default=None,
        metavar="NY",
        help="Same as --demo-cloth-panel-fixed-nx for ny (same vertex count per panel, not cell cm).",
    )
    parser.add_argument(
        "--demo-cloth-panel-rng-entropy",
        action="store_true",
        help=(
            "Sample each world's procedural cloth panel NumPy RNG seed from OS entropy at "
            "startup (different cloth geometry/stiffness each run; JSON still records the "
            "drawn seeds)."
        ),
    )


def add_demo_io_args(parser) -> None:
    """Optional toggles for USD export and metadata JSON (for benchmarking)."""
    parser.add_argument(
        "--demo-write-usd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --viewer usd: write each frame to the USD file; use --no-demo-write-usd to skip I/O.",
    )
    parser.add_argument(
        "--demo-write-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the run: write per-world metadata JSON; use --no-demo-write-json to skip.",
    )


def demo_scenario_from_args(args, base: DemoScenarioConfig | None = None) -> DemoScenarioConfig:
    """Merge ``args`` from :func:`add_demo_scenario_args` into a scenario config."""
    b = base or DEMO_SCENARIO
    wc_arg = getattr(args, "demo_world_count", None)
    world_count = b.world_count if wc_arg is None else wc_arg

    gr = getattr(args, "demo_grid_rows", None)
    gc = getattr(args, "demo_grid_cols", None)
    if gr is not None or gc is not None:
        if gr is None or gc is None:
            raise ValueError("Provide both --demo-grid-rows and --demo-grid-cols, or neither.")
        grid_rows, grid_cols = gr, gc
    elif wc_arg is not None and wc_arg != b.world_count:
        grid_rows, grid_cols = default_demo_grid_for_world_count(world_count)
    else:
        grid_rows, grid_cols = b.grid_rows, b.grid_cols

    if grid_rows * grid_cols != world_count:
        raise ValueError(
            f"grid_rows * grid_cols ({grid_rows * grid_cols}) must equal world_count ({world_count}); "
            "adjust --demo-grid-rows/--demo-grid-cols or --demo-world-count."
        )

    enable_visual_display = b.enable_visual_display
    if getattr(args, "demo_no_display", False):
        enable_visual_display = False

    write_usd = getattr(args, "demo_write_usd", b.write_usd)
    write_metadata_json = getattr(args, "demo_write_json", b.write_metadata_json)

    cloth_dens = getattr(args, "demo_cloth_grid_density", None)
    extra: dict = {}
    if cloth_dens is not None:
        extra["cloth_mesh_density_scale"] = float(cloth_dens)
    if getattr(args, "demo_cloth_panel_rng_entropy", False):
        extra["cloth_panel_rng_use_entropy"] = True
    fn = getattr(args, "demo_cloth_panel_fixed_nx", None)
    gn = getattr(args, "demo_cloth_panel_fixed_ny", None)
    _validate_fixed_cloth_panel_grid(fn, gn)
    if fn is not None:
        extra["cloth_panel_fixed_grid_nx"] = fn
        extra["cloth_panel_fixed_grid_ny"] = gn
    h = getattr(args, "demo_cloth_panel_target_cell_cm", None)
    if h is not None:
        x = float(h)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("--demo-cloth-panel-target-cell-cm must be a positive finite float.")
        extra["cloth_panel_target_cell_cm"] = x

    cfg = replace(
        b,
        world_count=world_count,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        enable_visual_display=enable_visual_display,
        write_usd=write_usd,
        write_metadata_json=write_metadata_json,
        **extra,
    )
    _validate_demo_cloth_panel_resolution(cfg)
    return cfg


def demo_arm_pool() -> list[DemoAssetSpec]:
    """Return candidate arm assets. Add more entries to randomize across worlds."""
    return [DemoAssetSpec(id="franka_panda", source="builtin:franka")]


def demo_animation_pool() -> list[DemoAssetSpec]:
    """Return candidate animation assets (ids only until external clips are wired)."""
    return [DemoAssetSpec(id="default_keyframe_track", source="builtin:robot_key_poses_in_example")]


def assign_demo_world_assets(
    world_count: int,
    arms: list[DemoAssetSpec],
    animations: list[DemoAssetSpec],
    *,
    seed: int | None,
) -> list[DemoWorldAssetAssignment]:
    """Assign one arm and one animation per world; reuse pool entries at random if needed."""
    if world_count < 1:
        raise ValueError("world_count must be >= 1")
    if not arms:
        raise ValueError("arm pool is empty")
    if not animations:
        raise ValueError("animation pool is empty")

    rng = np.random.default_rng(seed)
    arm_idx = rng.integers(0, len(arms), size=world_count)
    anim_idx = rng.integers(0, len(animations), size=world_count)

    out: list[DemoWorldAssetAssignment] = []
    for w in range(world_count):
        ai = int(arm_idx[w])
        gi = int(anim_idx[w])
        a = arms[ai]
        g = animations[gi]
        out.append(
            DemoWorldAssetAssignment(
                world_index=w,
                arm_id=a.id,
                arm_source=a.source,
                animation_id=g.id,
                animation_source=g.source,
                arm_pool_index=ai,
                animation_pool_index=gi,
            )
        )
    return out


DEMO_RECORDING = DemoRecordingPaths()


def add_demo_recording_args(parser) -> None:
    """Register CLI flags for metadata path (USD path stays ``--output-path``)."""
    parser.add_argument(
        "--demo-metadata-json",
        type=str,
        default=DEMO_RECORDING.metadata_json_path,
        help="Base path; one file per world is written as {name}_world_WWWW.json next to it.",
    )
    parser.add_argument(
        "--demo-output-root",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Parent directory only: each run creates DIR/run_<YYYYMMDD>_<HHMMSS>_<pid>/ with "
            "recording.usd, example_training_v1_4_meta.json, and per-world "
            "example_training_v1_4_meta_world_WWWW.json next to it (auto layout; safe for many "
            "parallel workers sharing DIR). Overrides --output-path and --demo-metadata-json."
        ),
    )


def demo_recording_paths_from_args(args) -> DemoRecordingPaths:
    """Combine viewer output path and demo metadata path from parsed CLI args."""
    usd = getattr(args, "output_path", None)
    meta = getattr(args, "demo_metadata_json", DEMO_RECORDING.metadata_json_path)
    return DemoRecordingPaths(usd_path=usd, metadata_json_path=meta)


def build_demo_cli_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by this example, including demo-specific flags."""
    parser = newton.examples.create_parser()
    add_demo_scenario_args(parser)
    add_demo_recording_args(parser)
    add_demo_io_args(parser)
    return parser


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)




@wp.kernel
def compute_ee_delta_batched(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_local_id: int,
    bodies_per_world: int,
    targets: wp.array(dtype=wp.transform),
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_local_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(targets[world_id])
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(targets[world_id])
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args, *, demo_config: DemoScenarioConfig | None = None):
        self.demo_config = demo_config if demo_config is not None else DEMO_SCENARIO
        cfg = self.demo_config
        _validate_demo_cloth_panel_resolution(cfg)
        if cfg.grid_rows * cfg.grid_cols != cfg.world_count:
            raise ValueError(
                f"grid_rows * grid_cols ({cfg.grid_rows * cfg.grid_cols}) must equal "
                f"world_count ({cfg.world_count})"
            )

        self.demo_recording_paths = demo_recording_paths_from_args(args) if args is not None else DEMO_RECORDING
        self._resolved_arm_pool = demo_arm_pool()
        self._resolved_animation_pool = demo_animation_pool()
        self.world_asset_assignments = assign_demo_world_assets(
            cfg.world_count,
            self._resolved_arm_pool,
            self._resolved_animation_pool,
            seed=cfg.asset_assignment_seed,
        )
        self.per_world_cloth_meta: list[dict] = []

        # simulation params
        self.num_envs = cfg.world_count
        self.grid_rows = cfg.grid_rows
        self.grid_cols = cfg.grid_cols
        self.cell_spacing_x = cfg.cell_spacing_x  # cm
        self.cell_spacing_y = cfg.cell_spacing_y  # cm

        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        # contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        # default cloth elasticity
        self.base_tri_ke = 1e4
        self.base_tri_ka = 1e4
        self.base_tri_kd = 1.5e-6
        self.base_bending_ke = 5.0
        self.base_bending_kd = 1e-2

        # Extra translation (cm) applied to every cloth panel after the default anchor
        # ``(cell_x + 20, cell_y - 50, 31.5)`` in :meth:`_make_panel_mesh`.
        # Keep this at zero for neutral training setup.
        self.cloth_initial_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self.scene = ModelBuilder(gravity=-981.0)
        self.viewer = viewer

        self._demo_fps_window_start = time.perf_counter()
        self._demo_fps_frame_count = 0

        self.world_offsets = self._build_world_offsets()

        if cfg.cloth_panel_rng_use_entropy and self.add_cloth:
            self._cloth_panel_numpy_seeds = [
                int.from_bytes(secrets.token_bytes(8), "little", signed=False)
                for _ in range(self.num_envs)
            ]
        else:
            self._cloth_panel_numpy_seeds = None

        if self.add_robot:
            self._add_robot_worlds()

        self.table_shape_indices = []
        self._add_tables()

        if self.add_cloth:
            self._add_panels()

        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # hide table primitive auto-rendering and draw them manually after scale conversion
        flags = self.model.shape_flags.numpy()
        for idx in self.table_shape_indices:
            flags[idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # meter-scale table visualization
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(p[0]) * self.viz_scale,
                        float(p[1]) * self.viz_scale,
                        float(p[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
                for p in self.table_positions
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (40.0 * self.viz_scale, 40.0 * self.viz_scale, 10.0 * self.viz_scale)
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5) for _ in range(self.num_envs)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.soft_contact_ke,
            )

        self.viewer.set_model(self.model)
        # Important: this example already places each env at a unique physical world-space
        # offset when building robots / tables / cloth. The viewer also supports its own
        # multi-world layout offsets, and set_model() auto-enables them when shape_world is
        # populated. For this example that causes the visual Franka meshes to be translated
        # a second time, so only one may remain in view. Disable viewer-side world offsets
        # and keep the original physical placement only.
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # wider camera for 5x5 layout
        center_x = 0.5 * (self.world_offsets[:, 0].min() + self.world_offsets[:, 0].max()) * self.viz_scale
        center_y = 0.5 * (self.world_offsets[:, 1].min() + self.world_offsets[:, 1].max()) * self.viz_scale
        self.viewer.set_camera(wp.vec3(center_x - 2.8, center_y + 1.8, 4.5), -20.0, -38.0)

        self.viz_state = self.model.state()

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._demo_print_scene_complexity()

        if isinstance(self.viewer, ViewerUSD) and not self.demo_config.write_usd:
            print(
                "[training_v1_4] ViewerUSD with write_usd=False: only begin/end_frame per tick "
                "(no log_state); output USD stays nearly empty.",
                flush=True,
            )

        if self.add_cloth:
            self.capture()

    def _demo_print_scene_complexity(self) -> None:
        """Print particle / mesh / graph sizes that dominate simulation cost."""
        m = self.model
        parts = [
            f"worlds={self.num_envs}",
            f"particles={m.particle_count}",
            f"edges={m.edge_count}",
            f"bodies={m.body_count}",
            f"shapes={m.shape_count}",
        ]
        if getattr(m, "spring_count", 0):
            parts.append(f"springs={m.spring_count}")
        line = "[training_v1_4] scene: " + " | ".join(parts)
        if self.add_cloth and self.per_world_cloth_meta:
            nxs = [int(meta["grid_nx"]) for meta in self.per_world_cloth_meta]
            nys = [int(meta["grid_ny"]) for meta in self.per_world_cloth_meta]
            verts = [nx * ny for nx, ny in zip(nxs, nys, strict=True)]
            tris = [2 * (nx - 1) * (ny - 1) for nx, ny in zip(nxs, nys, strict=True)]
            line += (
                f" | cloth panel res nx×ny: min ({min(nxs)},{min(nys)}) max ({max(nxs)},{max(nys)})"
                f" | cloth verts(total)={sum(verts)} tris(total)={sum(tris)}"
            )
        print(line, flush=True)

    def _build_world_offsets(self):
        offsets = []
        x0 = -0.5 * (self.grid_cols - 1) * self.cell_spacing_x
        y0 = -0.5 * (self.grid_rows - 1) * self.cell_spacing_y
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                offsets.append((x0 + c * self.cell_spacing_x, y0 + r * self.cell_spacing_y, 0.0))
        return np.array(offsets, dtype=np.float32)

    def _add_robot_worlds(self):
        self.endeffector_local_id = None
        self.table_positions = []

        for env_id in range(self.num_envs):
            builder = ModelBuilder()
            world_offset = self.world_offsets[env_id]
            assn = self.world_asset_assignments[env_id]
            arm_spec = DemoAssetSpec(id=assn.arm_id, source=assn.arm_source)
            self.create_articulation(builder, world_offset, arm_spec, env_id)
            self.scene.add_world(builder)

            if env_id == 0:
                self.bodies_per_world = builder.body_count
                self.dof_q_per_world = builder.joint_coord_count
                self.dof_qd_per_world = builder.joint_dof_count
                self.endeffector_local_id = builder.body_count - 3

    def _add_tables(self):
        self.table_positions = []
        for env_id in range(self.num_envs):
            ox, oy, oz = self.world_offsets[env_id]
            table_pos = wp.vec3(ox + 0.0, oy - 50.0, 10.0)
            self.table_positions.append(table_pos)
            shape_idx = self.scene.shape_count
            self.scene.add_shape_box(
                -1,
                wp.transform(table_pos, wp.quat_identity()),
                hx=40.0,
                hy=40.0,
                hz=10.0,
            )
            self.table_shape_indices.append(shape_idx)

    def _make_panel_mesh(self, env_id: int):
        if self.demo_config.cloth_panel_rng_use_entropy and self._cloth_panel_numpy_seeds is not None:
            cloth_rng_seed = int(self._cloth_panel_numpy_seeds[env_id])
        else:
            cloth_rng_seed = int(1000 + env_id)
        rng = np.random.default_rng(cloth_rng_seed)
        cfg = self.demo_config

        shape_mode = env_id % 3
        top_scale: float | None = None
        skew_cm: float | None = None

        width = float(rng.uniform(12.0, 56.0))
        height = float(rng.uniform(12.0, 56.0))

        (nx_lo_b, nx_hi), (ny_lo_b, ny_hi) = demo_scaled_cloth_grid_bounds(cfg)
        s = float(cfg.cloth_mesh_density_scale)

        target_cell_effective_cm: float | None = None
        if cfg.cloth_panel_target_cell_cm is not None:
            te = float(cfg.cloth_panel_target_cell_cm) / s
            target_cell_effective_cm = te
            nx = max(4, min(nx_hi, int(round(width / te)) + 1))
            ny = max(4, min(ny_hi, int(round(height / te)) + 1))
            nx_lo, nx_hi_meta, ny_lo, ny_hi_meta = nx, nx, ny, ny
        elif cfg.cloth_panel_fixed_grid_nx is not None:
            nx = max(4, int(round(cfg.cloth_panel_fixed_grid_nx * s)))
            ny = max(4, int(round(cfg.cloth_panel_fixed_grid_ny * s)))
            nx_lo, nx_hi_meta, ny_lo, ny_hi_meta = nx, nx, ny, ny
        else:
            nx = int(rng.integers(nx_lo_b, nx_hi + 1))
            ny = int(rng.integers(ny_lo_b, ny_hi + 1))
            nx_lo, nx_hi_meta = nx_lo_b, nx_hi
            ny_lo, ny_hi_meta = ny_lo_b, ny_hi

        if shape_mode == 0:  # rectangle
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-width * 0.5,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ width * 0.5,  height * 0.5, 0.0], dtype=np.float32)
        elif shape_mode == 1:  # trapezoid
            top_scale = float(rng.uniform(0.32, 0.96))
            top_w = width * top_scale
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-top_w * 0.5,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ top_w * 0.5,  height * 0.5, 0.0], dtype=np.float32)
        else:  # skewed quad
            skew_cm = float(rng.uniform(-0.48 * width, 0.48 * width))
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-width * 0.5 + skew_cm,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ width * 0.5 + skew_cm,  height * 0.5, 0.0], dtype=np.float32)

        ox, oy, _ = self.world_offsets[env_id]
        dx, dy, dz = self.cloth_initial_offset_cm
        target_center = np.array(
            [ox + 20.0 + dx, oy - 50.0 + dy, 31.5 + dz],
            dtype=np.float32,
        )
        panel_z_wave_amplitude_cm = 0.25

        verts = []
        for j in range(ny):
            v = j / (ny - 1)
            for i in range(nx):
                u = i / (nx - 1)
                p = (
                    (1.0 - u) * (1.0 - v) * c00
                    + u * (1.0 - v) * c10
                    + (1.0 - u) * v * c01
                    + u * v * c11
                )
                p[2] += panel_z_wave_amplitude_cm * math.sin(u * math.pi) * math.sin(v * math.pi)
                p += target_center
                verts.append(wp.vec3(float(p[0]), float(p[1]), float(p[2])))

        indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                indices.extend([a, b, d, a, d, c])

        tri_ke = self.base_tri_ke * float(rng.uniform(0.012, 28.0))
        tri_ka = self.base_tri_ka * float(rng.uniform(0.012, 28.0))
        edge_ke = self.base_bending_ke * float(rng.uniform(0.003, 55.0))
        tri_kd = self.base_tri_kd * float(rng.uniform(0.12, 25.0))
        edge_kd = self.base_bending_kd * float(rng.uniform(0.08, 40.0))
        density = float(rng.uniform(0.0012, 0.42))
        particle_radius = float(rng.uniform(0.22, 2.4))

        shape_names = ("rectangle", "trapezoid", "skewed_quad")
        cloth_meta = {
            "shape_mode": shape_names[shape_mode],
            "shape_mode_index": shape_mode,
            "panel_width_cm": width,
            "panel_height_cm": height,
            "grid_nx": nx,
            "grid_ny": ny,
            "tri_ke": tri_ke,
            "tri_ka": tri_ka,
            "tri_kd": tri_kd,
            "edge_ke": edge_ke,
            "edge_kd": edge_kd,
            "density": density,
            "cloth_panel_grid_fixed": cfg.cloth_panel_fixed_grid_nx is not None,
            "cloth_panel_fixed_grid_nx_config": cfg.cloth_panel_fixed_grid_nx,
            "cloth_panel_fixed_grid_ny_config": cfg.cloth_panel_fixed_grid_ny,
            "cloth_panel_target_cell_cm_config": cfg.cloth_panel_target_cell_cm,
            "cloth_panel_target_cell_effective_cm": target_cell_effective_cm,
            "cloth_panel_approx_mean_cell_width_cm": width / max(1, nx - 1),
            "cloth_panel_approx_mean_cell_height_cm": height / max(1, ny - 1),
            "particle_radius": particle_radius,
            "target_center_cm": target_center.tolist(),
            "target_center_offset_from_world_origin_cm": [20.0, -50.0, 31.5],
            "cloth_initial_offset_cm": list(self.cloth_initial_offset_cm),
            "panel_z_wave_amplitude_cm": panel_z_wave_amplitude_cm,
            "cloth_numpy_rng_seed": cloth_rng_seed,
            "top_scale": top_scale,
            "skew_cm": skew_cm,
            "base_tri_ke": self.base_tri_ke,
            "base_tri_ka": self.base_tri_ka,
            "base_bending_ke": self.base_bending_ke,
            "base_bending_kd": self.base_bending_kd,
            "cloth_mesh_density_scale": cfg.cloth_mesh_density_scale,
            "cloth_grid_nx_bounds_config": [cfg.cloth_grid_nx_min, cfg.cloth_grid_nx_max],
            "cloth_grid_ny_bounds_config": [cfg.cloth_grid_ny_min, cfg.cloth_grid_ny_max],
            "cloth_grid_nx_sample_bounds_effective": [nx_lo, nx_hi_meta],
            "cloth_grid_ny_sample_bounds_effective": [ny_lo, ny_hi_meta],
        }

        return verts, indices, tri_ke, tri_ka, tri_kd, edge_ke, edge_kd, density, particle_radius, cloth_meta

    def _add_panels(self):
        self.per_world_cloth_meta = []
        for env_id in range(self.num_envs):
            verts, indices, tri_ke, tri_ka, tri_kd, edge_ke, edge_kd, density, particle_radius, cloth_meta = (
                self._make_panel_mesh(env_id)
            )
            self.per_world_cloth_meta.append(cloth_meta)
            self.scene.add_cloth_mesh(
                vertices=verts,
                indices=indices,
                rot=wp.quat_identity(),
                pos=wp.vec3(0.0, 0.0, 0.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=density,
                scale=1.0,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                tri_kd=tri_kd,
                edge_ke=edge_ke,
                edge_kd=edge_kd,
                particle_radius=particle_radius,
            )

    def set_up_control(self):
        self.control = self.model.control()

        out_dim = 6
        in_dim = self.dof_qd_per_world

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_local_id: int,
            bodies_per_world: int,
            world_id: int,
            body_out: wp.array[float],
        ):
            body_id = bodies_per_world * world_id + body_local_id
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            x_wb = body_q[body_id]
            r_world = wp.transform_vector(x_wb, ee_offset - body_com[body_id])
            qd = body_qd[body_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(self.num_envs, dtype=wp.spatial_vector)

        self.initial_pose = self.model.joint_q.numpy().copy()
        self.initial_pose_per_world = self.initial_pose.reshape(self.num_envs, self.dof_q_per_world).copy()

        self.target_xforms = wp.empty(self.num_envs, dtype=wp.transform)
        self.target_xforms_np = np.zeros((self.num_envs, 7), dtype=np.float32)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder, world_offset, arm: DemoAssetSpec, env_id: int):
        if arm.source == "builtin:franka":
            asset_path = newton.utils.download_asset("franka_emika_panda")
            urdf_path = str(asset_path / "urdf" / "fr3_franka_hand.urdf")
        else:
            urdf_path = arm.source

        base_pos = (-50.0 + float(world_offset[0]), -50.0 + float(world_offset[1]), 0.0)
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(base_pos, wp.quat_identity()),
            floating=False,
            scale=100,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        if env_id == 0:
            clamp_close_activation_val = 0.1
            clamp_open_activation_val = 0.8

            self.robot_key_poses = np.array(
                [
                    [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                    [1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                    [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                    [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                    [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                ],
                dtype=np.float32,
            )
            self.targets_local = self.robot_key_poses[:, 1:]
            self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def compute_body_jacobian_for_world(self, model: Model, joint_q: wp.array, joint_qd: wp.array, world_id: int):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = self.dof_qd_per_world
        out_dim = 6
        qd_start = world_id * in_dim

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel,
                1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                    self.endeffector_local_id,
                    self.bodies_per_world,
                    world_id,
                ],
                outputs=[self.body_out],
            )

        J_np = np.zeros((out_dim, in_dim), dtype=np.float32)
        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            J_np[i, :] = joint_qd.grad.numpy()[qd_start : qd_start + in_dim]
            tape.zero()
        return J_np

    def generate_control_joint_qd(self, state_in: State):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return
        assert state_in.joint_q is not None
        assert state_in.joint_qd is not None

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        target_local = self.targets_local[current_interval]

        for env_id in range(self.num_envs):
            ox, oy, _ = self.world_offsets[env_id]
            self.target_xforms_np[env_id, 0] = target_local[0] + ox
            self.target_xforms_np[env_id, 1] = target_local[1] + oy
            self.target_xforms_np[env_id, 2] = target_local[2]
            self.target_xforms_np[env_id, 3:] = target_local[3:7]
        self.target_xforms = wp.array(self.target_xforms_np, dtype=wp.transform, device=self.model.device)

        wp.launch(
            compute_ee_delta_batched,
            dim=self.num_envs,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_local_id,
                self.bodies_per_world,
                self.target_xforms,
            ],
            outputs=[self.ee_delta],
        )

        q_all = state_in.joint_q.numpy()
        delta_target_all = self.ee_delta.numpy()
        target_joint_qd_np = np.zeros_like(state_in.joint_qd.numpy())

        for env_id in range(self.num_envs):
            q_start = env_id * self.dof_q_per_world
            qd_start = env_id * self.dof_qd_per_world

            q = q_all[q_start : q_start + self.dof_q_per_world]
            delta_target = delta_target_all[env_id]
            J = self.compute_body_jacobian_for_world(self.model, state_in.joint_q, state_in.joint_qd, env_id)
            J_inv = np.linalg.pinv(J)
            I = np.eye(J.shape[1], dtype=np.float32)
            N = I - J_inv @ J

            q_des = q.copy()
            q_des[1:] = self.initial_pose_per_world[env_id, 1:]

            K_null = 1.0
            delta_q_null = K_null * (q_des - q)
            delta_q = J_inv @ delta_target + N @ delta_q_null

            delta_q[-2] = target_local[-1] * 4.0 - q[-2]
            delta_q[-1] = target_local[-1] * 4.0 - q[-1]

            target_joint_qd_np[qd_start : qd_start + self.dof_qd_per_world] = delta_q

        self.target_joint_qd.assign(target_joint_qd_np)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        # Wall-clock FPS in :meth:`_demo_tick_print_fps` is meaningful only if GPU work
        # for this step has finished before the next frame is timed.
        wp.synchronize()

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                self.model.shape_contact_pair_count = 0
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _demo_tick_print_fps(self) -> None:
        """Once per main-loop iteration: print rolling FPS (GPU-synchronized wall time) every 1 s."""
        self._demo_fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._demo_fps_window_start
        if elapsed >= 1.0:
            fps = self._demo_fps_frame_count / elapsed
            print(
                f"[training_v1_4] loop FPS (GPU-sync wall): {fps:.2f}  "
                f"({self._demo_fps_frame_count} frames in {elapsed:.2f}s)"
                f"  sim_time={self.sim_time:.3f}s",
                flush=True,
            )
            self._demo_fps_window_start = now
            self._demo_fps_frame_count = 0

    def prepare_viz_state_for_viewer(self) -> None:
        """Copy simulation state into meter-scale buffers used for drawing.

        Fills :attr:`viz_state` from :attr:`state_0` (positions scaled by
        :attr:`viz_scale`; body rotations unchanged). Points :attr:`model`
        ``shape_transform`` / ``shape_scale`` at the precomputed meter-scale
        buffers so :meth:`~newton.viewer.ViewerBase.log_state` matches the GL
        path. Call :meth:`restore_sim_shape_buffers` after logging if simulation
        must continue in the same process.
        """
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

    def restore_sim_shape_buffers(self) -> None:
        """Restore :attr:`model` shape buffers to simulation (cm) space after display."""
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def render(self):
        self._demo_tick_print_fps()
        try:
            if not demo_should_run_render(self.viewer, self.demo_config.enable_visual_display):
                return

            if isinstance(self.viewer, ViewerUSD) and not self.demo_config.write_usd:
                self.viewer.begin_frame(self.sim_time)
                self.viewer.end_frame()
                return

            self.prepare_viz_state_for_viewer()

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.viz_state)
            self.viewer.log_shapes(
                "/tables", newton.GeoType.BOX, self.table_viz_scale, self.table_viz_xform, self.table_viz_color
            )
            self.viewer.end_frame()

            self.restore_sim_shape_buffers()
        finally:
            wp.synchronize()

    def _build_demo_sim_export_dict(self) -> dict:
        """Static simulation / solver fields duplicated into each world's sidecar JSON."""
        d = {
            "fps": self.fps,
            "frame_dt_s": self.frame_dt,
            "sim_substeps": self.sim_substeps,
            "sim_dt_s": self.sim_dt,
            "viz_scale_m_per_sim_unit": self.viz_scale,
            "sim_length_unit": "cm",
            "model_builder_gravity_cm_s2": -981.0,
            "add_cloth": self.add_cloth,
            "add_robot": self.add_robot,
            "cloth_particle_radius_cm": self.cloth_particle_radius,
            "cloth_body_contact_margin_cm": self.cloth_body_contact_margin,
            "particle_self_contact_radius_cm": self.particle_self_contact_radius,
            "particle_self_contact_margin_cm": self.particle_self_contact_margin,
            "soft_contact_ke": self.soft_contact_ke,
            "soft_contact_kd": self.soft_contact_kd,
            "self_contact_friction_mu": self.self_contact_friction,
            "robot_shape_material_ke": self.robot_contact_ke,
            "robot_shape_material_kd": self.robot_contact_kd,
            "robot_shape_material_mu": self.robot_contact_mu,
            "solver_featherstone_update_mass_matrix_interval": self.sim_substeps,
            "table_box_half_extents_cm": [40.0, 40.0, 10.0],
            "table_offset_from_world_origin_cm": [0.0, -50.0, 10.0],
        }
        if self.add_cloth:
            d["cloth_initial_offset_cm"] = list(self.cloth_initial_offset_cm)
        if self.cloth_solver is not None:
            d["solver_vbd"] = {
                "iterations": self.iterations,
                "integrate_with_external_rigid_solver": True,
                "particle_self_contact_radius_cm": self.particle_self_contact_radius,
                "particle_self_contact_margin_cm": self.particle_self_contact_margin,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius_cm": 0.5,
                "particle_enable_self_contact": True,
                "particle_vertex_contact_buffer_size": 16,
                "particle_edge_contact_buffer_size": 20,
                "particle_collision_detection_interval": -1,
                "rigid_contact_k_start": self.soft_contact_ke,
            }
        return d

    def build_demo_metadata_world_dict(self, world_index: int, *, metadata_json_path: Path | None = None) -> dict:
        """One self-contained sidecar dict for ``world_index`` (not including USD time samples)."""
        w = world_index
        if w < 0 or w >= self.num_envs:
            raise ValueError(f"world_index must be in [0, {self.num_envs}), got {w}")
        assn = self.world_asset_assignments[w]
        ox, oy, oz = float(self.world_offsets[w, 0]), float(self.world_offsets[w, 1]), float(self.world_offsets[w, 2])
        tp = self.table_positions[w]
        paths = self.demo_recording_paths
        this_json = metadata_json_path if metadata_json_path is not None else paths.per_world_metadata_path(w)

        arms = [{"id": a.id, "source": a.source} for a in self._resolved_arm_pool]
        anims = [{"id": a.id, "source": a.source} for a in self._resolved_animation_pool]

        robot_block: dict = {
            "control_scheme": "task_space_keyframes_jacobian_per_world",
            "endeffector_offset_cm": [0.0, 0.0, 22.0],
            "franka_initial_joint_q_first_6_rad": [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307],
            "robot_urdf_linear_scale": 100,
            "robot_base_offset_template_cm": [-50.0, -50.0, -10.0],
        }
        if hasattr(self, "robot_key_poses"):
            robot_block["robot_key_poses_table"] = self.robot_key_poses.tolist()
            robot_block["robot_key_poses_time_cumulative_s"] = np.cumsum(self.robot_key_poses[:, 0]).tolist()
        if self.endeffector_local_id is not None:
            robot_block["endeffector_body_index_local"] = self.endeffector_local_id
        if self.bodies_per_world is not None:
            robot_block["bodies_per_world"] = self.bodies_per_world
            robot_block["dof_q_per_world"] = self.dof_q_per_world
            robot_block["dof_qd_per_world"] = self.dof_qd_per_world

        cloth = self.per_world_cloth_meta[w] if self.add_cloth and w < len(self.per_world_cloth_meta) else None

        return {
            "schema": "newton.examples.training_v1_4/world/v1",
            "intent": "Rebuild procedural cloth + layout; pair with USD for full dynamic replay.",
            "world_index": w,
            "world_count": self.num_envs,
            "scenario": asdict(self.demo_config),
            "recording": {
                "usd_path": paths.usd_path,
                "metadata_json_this_world": str(this_json.resolve()),
            },
            "asset_pools_at_build": {"arms": arms, "animations": anims},
            "this_world": {
                "world_offset_cm": [ox, oy, oz],
                "table_center_cm": [float(tp[0]), float(tp[1]), float(tp[2])],
                "arm_id": assn.arm_id,
                "arm_source": assn.arm_source,
                "animation_id": assn.animation_id,
                "animation_source": assn.animation_source,
                "arm_pool_index": assn.arm_pool_index,
                "animation_pool_index": assn.animation_pool_index,
            },
            "layout_grid": {
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "cell_spacing_x_cm": self.cell_spacing_x,
                "cell_spacing_y_cm": self.cell_spacing_y,
            },
            "cloth": cloth,
            "sim": self._build_demo_sim_export_dict(),
            "robot": robot_block,
            "determinism_notes": {
                "cloth_rng": (
                    "numpy.random.Generator(PCG64); per-world seed from secrets.token_bytes(8) "
                    "at Example construction (cloth_panel_rng_use_entropy=True)"
                    if self.demo_config.cloth_panel_rng_use_entropy
                    else "numpy.random.Generator(PCG64) seed = 1000 + world_index"
                ),
                "asset_assignment_rng": (
                    f"numpy.random.Generator(PCG64) seed = {self.demo_config.asset_assignment_seed}"
                    if self.demo_config.asset_assignment_seed is not None
                    else "asset_assignment_seed is null (non-reproducible draws)"
                ),
                "gpu_replay": "Bit-identical physics may require same Warp/CUDA/device versions.",
            },
        }

    def build_demo_metadata_dict(self) -> dict:
        """Small manifest listing per-world sidecar paths (full data is in each world JSON)."""
        p = self.demo_recording_paths
        return {
            "schema": "newton.examples.training_v1_4/aggregate/v1",
            "world_count": self.num_envs,
            "usd_path": p.usd_path,
            "per_world_metadata_paths": [str(p.per_world_metadata_path(w).resolve()) for w in range(self.num_envs)],
        }


def write_demo_metadata(example: Example, paths: DemoRecordingPaths | None = None) -> list[Path]:
    """Write one sidecar JSON per world in the same directory as the metadata base path."""
    if not example.demo_config.write_metadata_json:
        print("[training_v1_4] skipped metadata JSON (write_metadata_json=False)", flush=True)
        return []
    paths = paths or example.demo_recording_paths
    paths.resolved_metadata_path().parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for w in range(example.num_envs):
        out = paths.per_world_metadata_path(w)
        payload = example.build_demo_metadata_world_dict(w, metadata_json_path=out)
        payload["recording"]["metadata_json_this_world"] = str(out.resolve())
        payload["recording"]["usd_path"] = paths.usd_path
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        written.append(out)
    if written:
        print(f"Demo metadata written ({len(written)} worlds): first {written[0]}")
    return written


def run_training_v1_4_demo(
    *,
    parser_defaults: dict[str, object] | None = None,
) -> Example:
    """Run the interactive / recording loop and write metadata JSON; return the example instance.

    Mirrors the module ``__main__`` block for programmatic / packaged use.

    Args:
        parser_defaults: Optional ``parser.set_defaults(**...)`` values (e.g. ``num_frames``).
    """
    parser = build_demo_cli_parser()
    if parser_defaults:
        parser.set_defaults(**parser_defaults)
    args_pre = parser.parse_args()
    apply_demo_recording_session_paths(args_pre)
    session_note = getattr(args_pre, "demo_output_session_dir", None)
    if session_note:
        print(f"[training_v1_4] recording session dir: {session_note}", flush=True)
        _saved_argv = sys.argv[:]
        try:
            sys.argv = _argv_with_demo_session_paths(
                _saved_argv,
                output_path=args_pre.output_path,
                metadata_json=args_pre.demo_metadata_json,
            )
            viewer, args = newton.examples.init(parser)
        finally:
            sys.argv = _saved_argv
    else:
        viewer, args = newton.examples.init(parser)
    demo_cfg = demo_scenario_from_args(args)
    example = Example(viewer, args, demo_config=demo_cfg)
    newton.examples.run(example, args)
    write_demo_metadata(example)
    return example


class TrainingDemoV1_4Api:
    """Explicit public surface for this module (vendor-friendly names).

    Wraps configuration, CLI registration, asset assignment, and metadata export.
    Prefer importing this class when exposing a stable API from a repackaged wheel.

    See ``TRAINING_V1_4_README.md`` (same directory) for CLI examples, programmatic
    entry points, how robotics teams can plug in custom URDFs (``demo_arm_pool``)
    versus animation metadata (``demo_animation_pool`` / future motion wiring),
    ``--demo-output-root`` session directories for parallel recording, and version
    notes at the end of ``TRAINING_V1_4_README.md``.

    Attributes:
        Simulation: Alias for :class:`Example` (the running batch simulation class).

    External display (e.g. host renderer): after each :meth:`Example.step`, call
    :meth:`Example.prepare_viz_state_for_viewer`, read :attr:`Example.viz_state`
    and :attr:`Example.model` (and table fields used in :meth:`Example.render`),
    then :meth:`Example.restore_sim_shape_buffers` before the next
    :meth:`Example.simulate` / :meth:`Example.step`.
    """

    ScenarioConfig = DemoScenarioConfig
    RecordingPaths = DemoRecordingPaths
    AssetSpec = DemoAssetSpec
    WorldAssetAssignment = DemoWorldAssetAssignment
    Simulation = Example

    make_scenario = staticmethod(make_demo_scenario)
    scenario_from_cli_args = staticmethod(demo_scenario_from_args)
    recording_paths_from_cli_args = staticmethod(demo_recording_paths_from_args)
    apply_recording_session_paths = staticmethod(apply_demo_recording_session_paths)
    assign_world_assets = staticmethod(assign_demo_world_assets)
    default_arm_pool = staticmethod(demo_arm_pool)
    default_animation_pool = staticmethod(demo_animation_pool)
    write_metadata_json = staticmethod(write_demo_metadata)
    build_cli_parser = staticmethod(build_demo_cli_parser)
    run_full_demo = staticmethod(run_training_v1_4_demo)
    default_grid_for_world_count = staticmethod(default_demo_grid_for_world_count)
    should_run_render = staticmethod(demo_should_run_render)
    viewer_writes_frame_log = staticmethod(demo_viewer_writes_frame_log)

    @staticmethod
    def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
        """Register ``--demo-*`` flags on an existing parser (after :func:`newton.examples.create_parser`)."""
        add_demo_scenario_args(parser)
        add_demo_recording_args(parser)
        add_demo_io_args(parser)


__all__ = [
    "TrainingDemoV1_4Api",
    "DEMO_RECORDING",
    "DEMO_SCENARIO",
    "DemoAssetSpec",
    "DemoRecordingPaths",
    "DemoScenarioConfig",
    "DemoWorldAssetAssignment",
    "Example",
    "add_demo_io_args",
    "add_demo_recording_args",
    "add_demo_scenario_args",
    "assign_demo_world_assets",
    "build_demo_cli_parser",
    "default_demo_grid_for_world_count",
    "demo_animation_pool",
    "demo_arm_pool",
    "apply_demo_recording_session_paths",
    "demo_recording_paths_from_args",
    "resolve_demo_recording_session_dir",
    "demo_scaled_cloth_grid_bounds",
    "demo_scenario_from_args",
    "demo_should_run_render",
    "demo_viewer_writes_frame_log",
    "make_demo_scenario",
    "run_training_v1_4_demo",
    "write_demo_metadata",
]


if __name__ == "__main__":
    run_training_v1_4_demo(parser_defaults={"num_frames": 3850})
