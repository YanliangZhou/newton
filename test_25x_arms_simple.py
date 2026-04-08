# Simple script to test 25 Franka arms visualization

import newton
from newton import ModelBuilder
import warp as wp
import numpy as np

class Test25xArms:
    def __init__(self, viewer):
        # simulation params
        self.num_envs = 25
        self.grid_rows = 5
        self.grid_cols = 5
        self.cell_spacing_x = 180.0  # cm
        self.cell_spacing_y = 180.0  # cm
        self.viz_scale = 0.01  # cm to m

        self.viewer = viewer
        self.scene = ModelBuilder(gravity=-981.0)
        self.world_offsets = self._build_world_offsets()
        self._add_robot_worlds()

        # Add ground plane
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Print debug information
        print(f"Number of bodies: {self.model.body_count}")
        print(f"Number of shapes: {self.model.shape_count}")
        print(f"Number of worlds: {self.model.world_count}")

        # Create visualization state
        self.viz_state = self.model.state()

        # Store original shape transforms and scales
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        # Create visualization versions with scale applied
        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Set model to viewer
        self.viewer.set_model(self.model)
        
        # Ensure all worlds are visible
        self.viewer.set_visible_worlds(None)
        
        # Set world offsets to zero to prevent double positioning
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # Set camera to top-down view to see all robots
        center_x = 0.5 * (self.world_offsets[:, 0].min() + self.world_offsets[:, 0].max()) * self.viz_scale
        center_y = 0.5 * (self.world_offsets[:, 1].min() + self.world_offsets[:, 1].max()) * self.viz_scale
        self.viewer.set_camera(wp.vec3(center_x, center_y, 10.0), -90.0, 0.0)
        print(f"Camera position: {center_x}, {center_y}, 10.0")

        self.sim_dt = 1.0 / 60.0
        self.sim_time = 0.0

    def _build_world_offsets(self):
        offsets = []
        x0 = -0.5 * (self.grid_cols - 1) * self.cell_spacing_x
        y0 = -0.5 * (self.grid_rows - 1) * self.cell_spacing_y
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x = x0 + c * self.cell_spacing_x
                y = y0 + r * self.cell_spacing_y
                offsets.append((x, y, 0.0))
        return np.array(offsets, dtype=np.float32)

    def _add_robot_worlds(self):
        self.bodies_per_world = 0
        for env_id in range(self.num_envs):
            builder = ModelBuilder()
            world_offset = self.world_offsets[env_id]
            self.create_articulation(builder, world_offset)
            self.scene.add_world(builder)

            if env_id == 0:
                self.bodies_per_world = builder.body_count
            print(f"env {env_id}: robot base = {world_offset[0]-50.0:.1f}, {world_offset[1]-50.0:.1f}, -10.0")

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

    def step(self):
        self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Update visualization state
        if self.model.body_count > 0:
            # Scale body transforms for visualization
            viz_body_q = wp.zeros_like(self.model.body_q)
            for i in range(self.model.body_count):
                q = self.model.body_q[i]
                viz_q = wp.transform(q.p * self.viz_scale, q.q)
                viz_body_q[i] = viz_q
            self.viz_state.body_q = viz_body_q

        # Update shape transforms for visualization
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        # Log state to viewer
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

        # Restore original transforms
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

if __name__ == "__main__":
    import newton.examples
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Test25xArms(viewer)
    newton.examples.run(example, args)
