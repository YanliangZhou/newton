# Simple test script to verify world assignments for Franka arms

import sys
import newton
from newton import ModelBuilder
import warp as wp

# Enable immediate output
sys.stdout.flush()

print("Starting simple world test...")
sys.stdout.flush()

# Create a builder for the main scene
main_builder = ModelBuilder(gravity=-981.0)

# Create 3 Franka arms, each in their own world
for i in range(3):
    print(f"Creating Franka arm {i}...")
    sys.stdout.flush()
    
    # Create a builder for this arm
    arm_builder = ModelBuilder()
    
    # Add Franka arm
    asset_path = newton.utils.download_asset("franka_emika_panda")
    base_pos = (i * 100.0, 0.0, -10.0)
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
    print(f"Added Franka arm {i} to world {i}")
    sys.stdout.flush()

# Add a ground plane (global world -1)
main_builder.add_ground_plane()
print("Added ground plane to global world")
sys.stdout.flush()

# Finalize the model
model = main_builder.finalize(requires_grad=False)
print("Model finalized")
sys.stdout.flush()

# Print model statistics
print("\n=== Model Statistics ===")
print(f"Number of worlds: {model.world_count}")
print(f"Number of bodies: {model.body_count}")
print(f"Number of shapes: {model.shape_count}")
sys.stdout.flush()

# Check shape world assignments
shape_worlds = model.shape_world.numpy()
print("\n=== Shape World Assignments ===")
print(f"Shape worlds: {shape_worlds}")
print(f"Unique worlds: {sorted(set(shape_worlds))}")
sys.stdout.flush()

# Check body world assignments
body_worlds = model.body_world.numpy()
print("\n=== Body World Assignments ===")
print(f"Body worlds: {body_worlds}")
print(f"Unique worlds: {sorted(set(body_worlds))}")
sys.stdout.flush()

print("\nTest completed!")
sys.stdout.flush()
