# Test script to verify 25 Franka arms are properly positioned

import newton
from newton import ModelBuilder
import warp as wp
import numpy as np

print("Starting 25x Franka arm position test...")

# Create a builder for the main scene
main_builder = ModelBuilder(gravity=-981.0)

# simulation params
num_envs = 25
grid_rows = 5
grid_cols = 5
cell_spacing_x = 180.0  # cm
cell_spacing_y = 180.0  # cm

# Build world offsets
world_offsets = []
x0 = -0.5 * (grid_cols - 1) * cell_spacing_x
y0 = -0.5 * (grid_rows - 1) * cell_spacing_y
for r in range(grid_rows):
    for c in range(grid_cols):
        world_offsets.append((x0 + c * cell_spacing_x, y0 + r * cell_spacing_y, 0.0))

# Create 25 Franka arms, each in their own world
bodies_per_world = 0
for i in range(num_envs):
    print(f"Creating Franka arm {i}...")
    
    # Create a builder for this arm
    arm_builder = ModelBuilder()
    
    # Add Franka arm
    asset_path = newton.utils.download_asset("franka_emika_panda")
    world_offset = world_offsets[i]
    base_pos = (-50.0 + float(world_offset[0]), -50.0 + float(world_offset[1]), -10.0)
    arm_builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(base_pos, wp.quat_identity()),
        floating=False,
        scale=100,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    
    # Add this arm as a new world
    main_builder.add_world(arm_builder)
    print(f"Added Franka arm {i} at position: {base_pos}")
    
    if i == 0:
        bodies_per_world = arm_builder.body_count

# Add a ground plane
main_builder.add_ground_plane()
print("Added ground plane")

# Finalize the model
model = main_builder.finalize(requires_grad=False)
print("Model finalized")

# Print model statistics
print("\n=== Model Statistics ===")
print(f"Number of worlds: {model.world_count}")
print(f"Number of bodies: {model.body_count}")
print(f"Expected bodies: {num_envs * bodies_per_world}")
print(f"Number of shapes: {model.shape_count}")

# Check body positions
print("\n=== Body Positions ===")
body_q = model.body_q.numpy()
for i in range(model.body_count):
    pos = body_q[i, :3]
    world = model.body_world.numpy()[i]
    if i % bodies_per_world == 0:  # Only print base positions
        print(f"Body {i} (World {world}): Position = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

print("\nTest completed!")
