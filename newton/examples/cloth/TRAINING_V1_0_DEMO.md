# Training scenario v1.0 (`training_v1_0`)

Multi-world batch: each world has a table, procedural cloth, and one manipulator from `demo_arm_pool()`, with optional USD recording and per-world JSON metadata.

**Module:** `newton.examples.cloth.example_training_v1_0` (file `example_training_v1_0.py`). The CLI short name is `training_v1_0` (Python module names cannot contain `.`, so version **v1.0** is encoded as **`v1_0`** in the filename).

The stable import surface is **`TrainingDemoV1Api`** in that module.

## Run from the CLI

All **command-line examples** below include **`--demo-cloth-panel-target-cell-cm`** (here `2.5` cm) so panel subdivisions track random panel size with roughly uniform cell spacing—adjust `H` for your training stack.

Examples are discovered from the filename `example_<name>.py`; this one is:

```text
training_v1_0
```

Typical invocation (recommended, uses `uv` so dependencies match the repo):

```bash
uv run --extra examples -m newton.examples training_v1_0 \
  --viewer usd \
  --output-path ./out/sim.usd \
  --demo-metadata-json ./out/meta.json \
  --demo-world-count 9 \
  --demo-grid-rows 3 \
  --demo-grid-cols 3 \
  --demo-cloth-panel-target-cell-cm 2.5 \
  --num-frames 1000
```

Same flags with the **`python`** launcher (after installing Newton into the active environment, for example `uv sync --extra examples` then `uv run python` or a venv’s `python`):

```bash
python -m newton.examples training_v1_0 \
  --viewer usd \
  --output-path ./out/sim.usd \
  --demo-metadata-json ./out/meta.json \
  --demo-world-count 9 \
  --demo-grid-rows 3 \
  --demo-grid-cols 3 \
  --demo-cloth-panel-target-cell-cm 2.5 \
  --num-frames 1000
```

On Windows **cmd.exe**, use caret line continuation instead of backslashes:

```bat
python -m newton.examples training_v1_0 ^
  --viewer usd ^
  --output-path .\out\sim.usd ^
  --demo-metadata-json .\out\meta.json ^
  --demo-world-count 9 ^
  --demo-grid-rows 3 ^
  --demo-grid-cols 3 ^
  --demo-cloth-panel-target-cell-cm 2.5 ^
  --num-frames 1000
```

Useful **`--demo-*`** flags (registered by `TrainingDemoV1Api.add_cli_arguments` / `build_demo_cli_parser`):

| Flag | Role |
|------|------|
| `--demo-world-count N` | Number of parallel worlds (must match grid rows × cols if you set both). |
| `--demo-grid-rows R` / `--demo-grid-cols C` | Layout; require `R*C == N`. |
| `--demo-no-display` | Skip interactive GL/Viser/Rerun; USD / file viewers still log. |
| `--demo-cloth-grid-density S` | Multiplies random `nx`/`ny` **bounds** from `cloth_grid_n*`, multiplies **fixed** `--demo-cloth-panel-fixed-nx/ny` bases, or **divides** the `--demo-cloth-panel-target-cell-cm` edge target (larger `S` → finer mesh). |
| `--demo-cloth-panel-target-cell-cm H` | **~Uniform physical mesh:** choose `nx`/`ny` from each panel’s random width/height so mean quad edge length ≈ `H` cm along each axis (large panels get more vertices). Capped by scaled `cloth_grid_*_max`. Cannot combine with fixed nx/ny. Command examples in this file use `H=2.5`. |
| `--demo-cloth-panel-fixed-nx NX` / `--demo-cloth-panel-fixed-ny NY` | **Same vertex count** on every panel: identical `nx`×`ny` (both required). Cell size in cm still changes with random panel size—use **target-cell** above if you need similar cell size instead. |
| `--demo-metadata-json PATH` | Base path for sidecar JSON (`*_world_0000.json`, …). |
| `--demo-write-usd` / `--no-demo-write-usd` | Toggle USD frame export when using `--viewer usd`. |
| `--demo-write-json` / `--no-demo-write-json` | Toggle metadata JSON at shutdown. |
| `--demo-cloth-panel-rng-entropy` | Per-world procedural cloth: draw NumPy RNG seeds from OS entropy at process start (different cloth each run; sidecar JSON still stores the realized seeds). Default is fixed `1000 + world_index` for repeatability. |

Standard Newton flags still apply (`--viewer`, `--output-path`, `--num-frames`, `--device`, …).

**Cloth resolution modes (pick one):** (1) default—random `nx`/`ny` in scaled bounds; (2) `--demo-cloth-panel-target-cell-cm`—similar **edge length in cm** across panels; (3) `--demo-cloth-panel-fixed-nx/ny`—same **vertex counts** everywhere. Trapezoid/skew panels are still only approximate for (2). **Runnable snippets in this file use (2) with `H=2.5`.** The stock `DEMO_SCENARIO` in code still omits (2) until you pass the flag or set the field.

## Programmatic API (short)

```python
import newton.examples.cloth.example_training_v1_0 as batch

cfg = batch.make_demo_scenario(16, cloth_mesh_density_scale=1.0, cloth_panel_target_cell_cm=2.5)
paths = batch.DemoRecordingPaths(usd_path="out.usd", metadata_json_path="meta.json")
# Or: batch.TrainingDemoV1Api.make_scenario(...)
```

`TrainingDemoV1Api` exposes aliases: `ScenarioConfig`, `AssetSpec`, `Simulation` (= `Example`), `run_full_demo`, `assign_world_assets`, etc.

---

## For robotics teams: own arm URDFs and animation assets

### 1. Custom manipulator URDFs (supported today)

Arm candidates come from **`demo_arm_pool()`** in the same module. Each entry is a **`DemoAssetSpec`**:

- **`id`**: stable string stored in JSON (your asset name).
- **`source`**: either **`"builtin:franka"`** (downloads bundled Franka) or an **absolute filesystem path** to a `.urdf` file.

Edit `demo_arm_pool()` to return your pool, for example:

```python
def demo_arm_pool() -> list[DemoAssetSpec]:
    return [
        DemoAssetSpec(id="acme_arm_a", source=r"C:\assets\robots\acme_arm\robot.urdf"),
        DemoAssetSpec(id="acme_arm_b", source="/mnt/robot_assets/acme_arm_b/model.urdf"),
        DemoAssetSpec(id="franka_lab", source="builtin:franka"),
    ]
```

**Assignment:** `assign_demo_world_assets` draws one arm per world from this list (with replacement if the pool is shorter than `world_count`). Reproducibility is controlled by **`DemoScenarioConfig.asset_assignment_seed`** (CLI does not override it today; set it in code or replace `DEMO_SCENARIO`). Cloth panel randomness is separate: use **`DemoScenarioConfig.cloth_panel_rng_use_entropy`** or **`--demo-cloth-panel-rng-entropy`** so each training run does not reuse the same default cloth seeds (`1000 + world_index`).

After this edit, run the **same** `python -m newton.examples training_v1_0 ...` command as above; there is **no** separate `--arm-urdf` flag in the stock example.

**Important simulation caveats (read before production use):**

- **`create_articulation`** loads non-builtin URDFs with **`scale=100`** (URDF meters → scene centimeters), same as Franka.
- **Initial joint pose** is still the Franka-specific first six values; for other arms you will likely need to change that block.
- **End-effector control** assumes a compatible last link / gripper offset (`endeffector_local_id`, `endeffector_offset`); adjust for your chain.
- **Jacobian reuse:** the example computes **one** manipulator Jacobian from **world 0** and applies it to every world. That is only valid when all arms share the **same kinematics** and stay in a **similar configuration**. If each world loads a **different** URDF, you must change the control code to compute **per-world** Jacobians.

### 2. Custom “animation” assets (metadata today, motion wiring is separate)

**`demo_animation_pool()`** returns **`DemoAssetSpec`** entries whose `id` / `source` strings are written into **per-world JSON** and drive **which asset id** was assigned. The **actual Cartesian trajectory** in the current example is still the **hard-coded** `robot_key_poses` table inside **`create_articulation`** (world 0). There is **no** CLI flag to point at an external trajectory file yet.

So for robotics teams today:

- Use **`demo_animation_pool()`** to document **which clip / policy / motion library** each world is *intended* to use (for provenance and downstream tooling).
- To **really** drive the sim from your own motion, fork or extend the example: branch on `animation_id` / `animation_source` (from `world_asset_assignments[env_id]`) when building targets or joint references, and replace the fixed keyframe array.

Example pool for metadata + future wiring:

```python
def demo_animation_pool() -> list[DemoAssetSpec]:
    return [
        DemoAssetSpec(id="pick_place_v3", source="file:///mnt/mocap/pick_place_v3.npz"),
        DemoAssetSpec(id="fold_slow", source="https://internal.example/anim/fold.json"),
        DemoAssetSpec(id="default_keyframe_track", source="builtin:robot_key_poses_in_example"),
    ]
```

### 3. Command-line recipe (after editing pools in Python)

Typical **headless USD + JSON** batch for your assets (Windows PowerShell style):

```powershell
uv run --extra examples -m newton.examples training_v1_0 `
  --viewer usd `
  --output-path "C:\data\batch\sim.usd" `
  --demo-metadata-json "C:\data\batch\meta.json" `
  --demo-world-count 9 `
  --demo-grid-rows 3 `
  --demo-grid-cols 3 `
  --demo-no-display `
  --demo-cloth-panel-target-cell-cm 2.5 `
  --demo-cloth-grid-density 0.6 `
  --num-frames 2000
```

Equivalent using **`python -m`** (same venv / install as above):

```powershell
python -m newton.examples training_v1_0 `
  --viewer usd `
  --output-path "C:\data\batch\sim.usd" `
  --demo-metadata-json "C:\data\batch\meta.json" `
  --demo-world-count 9 `
  --demo-grid-rows 3 `
  --demo-grid-cols 3 `
  --demo-no-display `
  --demo-cloth-panel-target-cell-cm 2.5 `
  --demo-cloth-grid-density 0.6 `
  --demo-cloth-panel-rng-entropy `
  --num-frames 2000
```

Paths in **`demo_arm_pool()`** must be valid on the machine that runs this command.

---

## Metadata layout (brief)

- JSON schemas use identifiers such as `newton.examples.training_v1_0/world/v2` and `.../aggregate/v1`.
- Default base metadata filename in code: `example_training_v1_0_meta.json` (override with `--demo-metadata-json`).
- Each world file lists **`scenario`** (full `DemoScenarioConfig` as dict), **`this_world`** (offsets, table, assigned **`arm_id` / `animation_id`** and pool indices), **`cloth`**, **`robot`**, **`sim`**, etc.

Use these JSON files together with USD for replay or dataset documentation.
