print("Testing single Franka arm...")

import newton
from newton import ModelBuilder
import warp as wp

# Create a builder for the main scene
main_builder = ModelBuilder(gravity=-981.0)

# Create a single Franka arm
print("Creating Franka arm...")
arm_builder = ModelBuilder()

# Add Franka arm
asset_path = newton.utils.download_asset("franka_emika_panda")
base_pos = (0.0, 0.0, -10.0)
print(f"Arm base position = ({base_pos[0]:.1f}, {base_pos[1]:.1f}, {base_pos[2]:.1f})")

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
print("Added arm to world 0")

# Finalize the model
print("Finalizing model...")
model = main_builder.finalize(requires_grad=False)
print("Model finalized")

# Print model statistics
print("=== Model Statistics ===")
print(f"Number of worlds: {model.world_count}")
print(f"Number of bodies: {model.body_count}")
print(f"Number of shapes: {model.shape_count}")

print("Test completed!")
