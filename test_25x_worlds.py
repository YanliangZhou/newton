# Test script to verify 25 Franka arms are properly added with correct world assignments

import sys
import newton
import newton.examples
from newton import ModelBuilder
import warp as wp

# Enable immediate output
sys.stdout.flush()

class TestExample:
    def __init__(self):
        # simulation params
        self.num_envs = 25
        self.grid_rows = 5
        self.grid_cols = 5
        self.cell_spacing_x = 180.0  # cm
        self.cell_spacing_y = 180.0  # cm

        self.scene = ModelBuilder(gravity=-981.0)
        self.world_offsets = self._build_world_offsets()
        self._add_robot_worlds()
        self._add_tables()

        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        print("=== Test Results ===")
        print(f"Number of worlds: {self.model.world_count}")
        print(f"Expected worlds: {self.num_envs}")
        print(f"Number of bodies: {self.model.body_count}")
        print(f"Bodies per world: {self.bodies_per_world}")
        print(f"Expected bodies: {self.num_envs * self.bodies_per_world}")
        print(f"Number of shapes: {self.model.shape_count}")

        # Check shape world assignments
        shape_worlds = self.model.shape_world.numpy()
        unique_worlds = set(shape_worlds)
        print(f"\nUnique shape worlds: {sorted(unique_worlds)}")
        print(f"Number of unique worlds: {len(unique_worlds)}")

        # Count shapes per world
        world_shape_count = {}
        for world in shape_worlds:
            if world not in world_shape_count:
                world_shape_count[world] = 0
            world_shape_count[world] += 1

        print("\nShapes per world:")
        for world in sorted(world_shape_count.keys()):
            print(f"World {world}: {world_shape_count[world]} shapes")

        # Check body world assignments
        body_worlds = self.model.body_world.numpy()
        unique_body_worlds = set(body_worlds)
        print(f"\nUnique body worlds: {sorted(unique_body_worlds)}")
        print(f"Number of unique body worlds: {len(unique_body_worlds)}")

        # Count bodies per world
        world_body_count = {}
        for world in body_worlds:
            if world not in world_body_count:
                world_body_count[world] = 0
            world_body_count[world] += 1

        print("\nBodies per world:")
        for world in sorted(world_body_count.keys()):
            print(f"World {world}: {world_body_count[world]} bodies")

    def _build_world_offsets(self):
        offsets = []
        x0 = -0.5 * (self.grid_cols - 1) * self.cell_spacing_x
        y0 = -0.5 * (self.grid_rows - 1) * self.cell_spacing_y
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                offsets.append((x0 + c * self.cell_spacing_x, y0 + r * self.cell_spacing_y, 0.0))
        return offsets

    def _add_robot_worlds(self):
        self.endeffector_local_id = None
        self.table_positions = []

        for env_id in range(self.num_envs):
            builder = ModelBuilder()
            world_offset = self.world_offsets[env_id]
            self.create_articulation(builder, world_offset)
            self.scene.add_world(builder)

            if env_id == 0:
                self.bodies_per_world = builder.body_count
                self.dof_q_per_world = builder.joint_coord_count
                self.dof_qd_per_world = builder.joint_dof_count
                self.endeffector_local_id = builder.body_count - 3
            print(f"env {env_id}: robot base = {world_offset[0]-50.0:.1f}, {world_offset[1]-50.0:.1f}, -10.0")

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

    def create_articulation(self, builder, world_offset):
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
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

if __name__ == "__main__":
    print("Starting test...")
    test = TestExample()
    print("Test completed.")
