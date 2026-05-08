# Training v1.4 (`training_v1_4`)

Multi-world cloth + arm batch demo; optional USD + per-world JSON. **Module:** `newton.examples.cloth.example_training_v1_4` · **CLI:** `python -m newton.examples training_v1_4` · **API:** `TrainingDemoV1_4Api`. **v1.4** is meant to drive **Isaac Sim**, another DCC, or any custom viewer from the same simulation outputs—not only Newton’s GL window.

Dependencies: `uv sync --extra examples`. Run from repo root `newton/`.

```bash
python -m newton.examples training_v1_4 \
  --viewer gl \
  --demo-world-count 9 --demo-grid-rows 3 --demo-grid-cols 3 \
  --demo-cloth-panel-target-cell-cm 2.5 \
  --num-frames 1000
```

`--viewer usd --demo-output-root ./runs` writes headless USD + JSON (no GL). `--demo-output-root` → `runs/run_<date>_<time>_<pid>/recording.usd`, `example_training_v1_4_meta.json`, `example_training_v1_4_meta_world_*.json`. Fixed paths: `--output-path` + `--demo-metadata-json`, omit `--demo-output-root`.

Upstream `training_v1_2` stays in `example_training_v1_2.py` (`TRAINING_V1_2_README.md`).

## `--demo-*` flags

| Flag | Purpose |
|------|---------|
| `--demo-world-count` | World count; with rows/cols, `R×C == N`. |
| `--demo-grid-rows` / `--demo-grid-cols` | Grid layout. |
| `--demo-no-display` | No GL/Viser/Rerun; USD/file recording unchanged. |
| `--demo-output-root` | Session folder layout; overrides `--output-path` / `--demo-metadata-json`. |
| `--demo-metadata-json` | Base JSON; per-world `{stem}_world_{i:04d}{suffix}`. |
| `--demo-cloth-panel-target-cell-cm` | ~uniform quad edge length (cm); mutually exclusive with fixed nx/ny. |
| `--demo-cloth-panel-fixed-nx/ny` | Same grid both axes (both required). |
| `--demo-cloth-grid-density` | Scales random nx/ny bounds or divides target cell size. |
| `--demo-cloth-panel-rng-entropy` | Panel RNG from OS entropy (non-reproducible); default `1000 + world_index`. |
| `--demo-write-usd` / `--no-demo-write-usd` | USD frames when `--viewer usd`. |
| `--demo-write-json` / `--no-demo-write-json` | Per-world JSON at exit. |

Also: `--viewer`, `--demo-no-display`, `--num-frames`, `--device`, `--benchmark`, …

**Cloth resolution:** default random nx/ny; or `--demo-cloth-panel-target-cell-cm`; or fixed nx/ny pair.

## Programmatic

```python
import newton.examples.cloth.example_training_v1_4 as t

cfg = t.make_demo_scenario(16, cloth_panel_target_cell_cm=2.5)
# TrainingDemoV1_4Api: ScenarioConfig, Simulation, run_full_demo, …
```

## External display (Isaac Sim, other hosts, no Newton viewer)

Use this when physics runs in Newton and **rendering lives in Isaac Sim** (or any other app): after each `Example.step()`, `prepare_viz_state_for_viewer()` → read `viz_state` (`particle_q`, `body_q`, **m**), `table_viz_*`, topology from `model` → **`restore_sim_shape_buffers()`** before the next step. Sim stays **cm**; display buffers use `viz_scale` (typically `0.01`). Skip `render()` on the Newton side.

## Arms / animations / metadata

- Arms: `demo_arm_pool()` (`DemoAssetSpec`, URDF path or `builtin:franka`); assignment seed `DemoScenarioConfig.asset_assignment_seed`.
- Motion still from `robot_key_poses` unless you branch on `animation_id`.
- JSON schemas: `newton.examples.training_v1_4/world/v1`, `…/aggregate/v1`.

---

## Version notes

**v1.4** — **Isaac Sim / other viewports:** same display snapshot contract as GL (prepare → read → restore) so you can wire Newton into Isaac or any host without forking `render()` or leaving collision shapes in the wrong length unit. Loop FPS is GPU-synchronized.

**v1.3** — Per-world Jacobian (not world-0 reused everywhere); EE error matches `cloth_franka`; stripped demo-only trajectory/layout offsets.

**v1.2** — Multi-world batch in one model; Jacobian from world 0 reused for speed—unsafe if URDFs or poses diverge (`TRAINING_V1_2_README.md`).

**v1.1** — Fixed cloth mesh density for fair VBD comparisons; repeatable seeds vs entropy for panels.
