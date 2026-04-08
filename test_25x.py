import newton
from newton.examples.cloth.example_cloth_franka_25x import Example
import newton.examples

# Create parser and viewer in headless mode
parser = newton.examples.create_parser()
viewer, args = newton.examples.init(parser, headless=True)

# Create example
example = Example(viewer, args)

# Print debug information
print(f"Number of bodies: {example.model.body_count}")
print(f"Bodies per world: {example.bodies_per_world}")
print(f"Expected bodies: {example.num_envs * example.bodies_per_world}")
print(f"Number of shapes: {example.model.shape_count}")
print(f"Number of worlds: {example.num_envs}")
print(f"World offsets shape: {example.world_offsets.shape}")

# Check if all worlds have been added
if example.model.body_count == example.num_envs * example.bodies_per_world:
    print("All worlds have been added successfully!")
else:
    print("Error: Not all worlds have been added!")
