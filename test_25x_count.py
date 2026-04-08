# Simple test script to verify 25 Franka arms are created

import newton
from newton import ModelBuilder
import warp as wp

print("Testing 25x Franka arm creation...")

# Create a builder for the main scene
main_builder = ModelBuilder(gravity=-981.0)

# Create 25 Franka arms
for i in range(25):
    print(f"Creating Franka arm {i}...")
    arm_builder = ModelBuilder()
    
    # Add Franka arm
    asset_path = newton.utils.download_asset("franka_emika_panda")
    arm_builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform((0, 0, 0), wp.quat_identity()),
        floating=False,
        scale=100,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    
    # Add this arm as a new world
    main_builder.add_world(arm_builder)

# Finalize the model
model = main_builder.finalize(requires_grad=False)

print("\n=== Results ===")
print(f"Number of worlds: {model.world_count}")
print(f"Number of bodies: {model.body_count}")
print(f"Number of shapes: {model.shape_count}")

print("\nTest completed!")
