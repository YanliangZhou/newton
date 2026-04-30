# Training v1.2 (`training_v1_2`)

Batch of independent worlds (table + procedural cloth + arm per world), optional USD + per-world JSON. **Module:** `newton.examples.cloth.example_training_v1_2` · **CLI:** `python -m newton.examples training_v1_2` · **Stable API:** `TrainingDemoV1_2Api` (same module). Filename uses `v1_2` because Python modules cannot contain `.`.

## Quick run

Dependencies: `uv sync --extra examples` (or equivalent install).

File placement (if you share only these two files):

- Put `example_training_v1_2.py` in `newton/examples/cloth/`
- Put `TRAINING_V1_2_README.md` in `newton/examples/cloth/`
- Then run commands from repository root `newton/`

There are two common viewer modes:

- `gl`: opens an interactive window and displays cloth + robot. No USD/JSON files are written by default.
- `usd`: headless recording mode. It does not open a GL window, and writes USD + JSON outputs.

For `gl`, run:

```bash
python -m newton.examples training_v1_2 \
  --viewer gl \
  --demo-world-count 9 --demo-grid-rows 3 --demo-grid-cols 3 \
  --demo-cloth-panel-target-cell-cm 2.5 \
  --num-frames 1000
```

For `usd`, run:

```bash
python -m newton.examples training_v1_2 \
  --viewer usd \
  --demo-output-root ./runs \
  --demo-world-count 9 --demo-grid-rows 3 --demo-grid-cols 3 \
  --demo-cloth-panel-target-cell-cm 2.5 \
  --num-frames 1000
```

Run the commands from the repository root (`newton/`) so Python can resolve `newton.examples`.

`--demo-output-root` creates `runs/run_<YYYYMMDD>_<HHMMSS>_<pid>/` with `recording.usd`, `example_training_v1_2_meta.json`, and `example_training_v1_2_meta_world_*.json` (see below). Startup logs `[training_v1_2] recording session dir: …`.

**Fixed paths instead** (single process; you avoid collisions yourself): set `--output-path out.usd` and `--demo-metadata-json out/meta.json` and omit `--demo-output-root`.

Use `python -m newton.examples training_v1_2` with the same flags if Newton is already on `PYTHONPATH`. On Windows **cmd.exe**, use `^` line continuation or a single line.

The previous CLI/module names `training_v1_1` / `example_training_v1_1` remain as **deprecated** aliases (see CHANGELOG).

## Recording layout

With `--demo-output-root DIR`:

```text
DIR/run_<YYYYMMDD>_<HHMMSS>_<pid>/
  recording.usd
  example_training_v1_2_meta.json
  example_training_v1_2_meta_world_0000.json
  …
```

One folder per OS process run; timestamp + pid avoids clashes when many workers share the same `DIR`. **Do not** point several processes at the same `--output-path` USD file.

## `--demo-*` flags

| Flag | Purpose |
|------|---------|
| `--demo-world-count N` | Worlds; with `--demo-grid-rows/cols`, require `R*C == N`. |
| `--demo-grid-rows` / `--demo-grid-cols` | Grid layout. |
| `--demo-no-display` | No GL/Viser/Rerun; USD/file viewers still record. |
| `--demo-output-root DIR` | Session dir layout above; overrides `--output-path` and `--demo-metadata-json`. |
| `--demo-metadata-json PATH` | Base JSON path; per-world files are `{stem}_world_{i:04d}{suffix}` alongside it. |
| `--demo-cloth-panel-target-cell-cm H` | ~Uniform quad edge length (cm) from random panel size (mutually exclusive with fixed nx/ny). |
| `--demo-cloth-panel-fixed-nx` / `--demo-cloth-panel-fixed-ny` | Same `nx`×`ny` on every panel (both required). |
| `--demo-cloth-grid-density S` | Scales random `nx`/`ny` bounds, fixed-grid bases, or divides target-cell edge length. |
| `--demo-cloth-panel-rng-entropy` | Cloth panel NumPy seeds from OS entropy (non-reproducible); default `1000 + world_index`. |
| `--demo-write-usd` / `--no-demo-write-usd` | USD frame export when `--viewer usd`. |
| `--demo-write-json` / `--no-demo-write-json` | Per-world JSON at shutdown. |

Other Newton flags: `--viewer`, `--num-frames`, `--device`, `--benchmark`, …

## Cloth resolution (pick one)

1. **Default** — random `nx`/`ny` within scaled config bounds.  
2. **`--demo-cloth-panel-target-cell-cm`** — mean edge spacing ≈ `H` cm per axis.  
3. **`--demo-cloth-panel-fixed-nx/ny`** — identical vertex counts on all panels.

`DEMO_SCENARIO` in code does not set (2) unless you pass the flag or build config in code.

## Programmatic use

```python
import newton.examples.cloth.example_training_v1_2 as t

cfg = t.make_demo_scenario(16, cloth_panel_target_cell_cm=2.5)
# t.TrainingDemoV1_2Api: ScenarioConfig, Simulation, run_full_demo,
# apply_recording_session_paths (on argparse.Namespace before init), …
```

## Custom arms and “animations”

**Arms:** edit `demo_arm_pool()` → list of `DemoAssetSpec(id=..., source=...)` with `source` either `"builtin:franka"` or a filesystem path to `.urdf`. `assign_demo_world_assets` samples one per world; **`asset_assignment_seed`** on `DemoScenarioConfig` controls draws (no dedicated CLI flag).

**Caveats:** URDFs use `scale=100` (m→cm); initial `joint_q` and Jacobian EE logic are Franka-oriented; **Jacobian is computed from world 0 only** and reused on all worlds — valid only for the same kinematic chain and similar `q`. Different URDFs per world need per-world Jacobians in code.

**Animations:** `demo_animation_pool()` ids/sources are written to JSON for provenance. Motion is still the hard-coded `robot_key_poses` in `create_articulation`; drive from your data by branching on `animation_id` / `world_asset_assignments` in a fork.

## Metadata

Schemas include `newton.examples.training_v1_2/world/v3` and `.../aggregate/v2`. Each world JSON holds `scenario`, `this_world`, `cloth`, `robot`, `sim`, `recording` (paths to USD and that JSON). Pair with USD for replay or dataset docs.

---

## Version notes

Training scenario v1.2 (latest)

- Reported FPS reflects wall time after GPU work completes, not overlapping async launches alone.
- Exports and versioning move to the v1.2 module and schema; outputs are organized so parallel runs do not collide, with the old entry point kept only as a deprecated alias.

Training scenario v1.1

- Cloth density was fixed (not randomized per panel) so VBD stays easier to compare across cloth mesh resolutions.
- Panel randomness can draw from OS entropy for varied rollouts, while a deterministic default remains available when you need repeatability.
