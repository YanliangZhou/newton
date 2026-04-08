import newton
from newton import ModelBuilder, State
import numpy as np
import warp as wp

# Create a scene
scene = ModelBuilder(gravity=-981.0)

# Add 25 robots
num_envs = 25
grid_rows = 5
grid_cols = 5
cell_spacing_x = 180.0  # cm
cell_spacing_y = 180.0  # cm

# Build world offsets
offsets = []
x0 = -0.5 * (grid_cols - 1) * cell_spacing_x
y0 = -0.5 * (grid_rows - 1) * cell_spacing_y
for r in range(grid_rows):
    for c in range(grid_cols):
        offsets.append((x0 + c * cell_spacing_x, y0 + r * cell_spacing_y, 0.0))
world_offsets = np.array(offsets, dtype=np.float32)

# Add robots
bodies_per_world = 0
for env_id in range(num_envs):
    builder = ModelBuilder()
    world_offset = world_offsets[env_id]
    
    # Add URDF
    asset_path = newton.utils.download_asset("franka_emika_panda")
    base_pos = (-50.0 + float(world_offset[0]), -50.0 + float(world_offset[1]), -10.0)
    builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(base_pos, wp.quat_identity()),
        floating=False,
        scale=100,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    
    scene.add_world(builder)
    
    if env_id == 0:
        bodies_per_world = builder.body_count
    
    print(f"Added robot {env_id} at position: {base_pos}")

# Finalize model
model = scene.finalize(requires_grad=False)

# Print results
print(f"\nFinal model:")
print(f"Number of bodies: {model.body_count}")
print(f"Bodies per world: {bodies_per_world}")
print(f"Expected bodies: {num_envs * bodies_per_world}")
print(f"Number of shapes: {model.shape_count}")

if model.body_count == num_envs * bodies_per_world:
    print("All robots have been added successfully!")
else:
    print("Error: Not all robots have been added!")
