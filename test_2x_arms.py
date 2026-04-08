# Simple script to test 2 Franka arms visualization

import newton
from newton import ModelBuilder
import warp as wp

class Test2xArms:
    def __init__(self, viewer):
        self.viewer = viewer
        self.scene = ModelBuilder(gravity=-981.0)
        self.viz_scale = 0.01  # cm to m

        # Create first Franka arm
        print("Creating first Franka arm...")
        arm1_builder = ModelBuilder()
        asset_path = newton.utils.download_asset("franka_emika_panda")
        arm1_builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-100.0, 0.0, -10.0), wp.quat_identity()),
            floating=False,
            scale=100,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.scene.add_world(arm1_builder)

        # Create second Franka arm
        print("Creating second Franka arm...")
        arm2_builder = ModelBuilder()
        arm2_builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((100.0, 0.0, -10.0), wp.quat_identity()),
            floating=False,
            scale=100,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.scene.add_world(arm2_builder)

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

        # Set camera position to see both robots
        self.viewer.set_camera(wp.vec3(0.0, 0.0, 5.0), -90.0, 0.0)
        print("Camera position: 0.0, 0.0, 5.0")

        self.sim_dt = 1.0 / 60.0
        self.sim_time = 0.0

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

    example = Test2xArms(viewer)
    newton.examples.run(example, args)
