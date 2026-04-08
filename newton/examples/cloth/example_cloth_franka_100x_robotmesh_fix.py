# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# 100x Cloth Franka Batch
#
# Based on example_cloth_franka.py, but creates 100 independent copies
# of the Franka + table + cloth scene in a 5x5 grid and runs them together.
#
# Main changes:
#   - 100 Franka articulations, one per world
#   - 100 tables
#   - 100 procedurally generated panels (different shape / size / stiffness)
#   - batched end-effector target computation for all worlds
#
# Notes:
#   - Simulation still uses centimeter scale, like the original example.
#   - The robots share the same nominal motion pattern, offset into each cell.
#   - Jacobian is computed from world 0 and reused across worlds. This is
#     usually fine when all robots start from the same configuration and
#     follow the same controller structure.
###########################################################################

from __future__ import annotations

import math
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)




@wp.kernel
def compute_ee_delta_batched(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_local_id: int,
    bodies_per_world: int,
    targets: wp.array(dtype=wp.transform),
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_local_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(targets[world_id])
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(targets[world_id])
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args):
        # simulation params
        self.num_envs = 100
        self.grid_rows = 10
        self.grid_cols = 10
        self.cell_spacing_x = 180.0  # cm
        self.cell_spacing_y = 180.0  # cm

        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        # contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        # default cloth elasticity
        self.base_tri_ke = 1e4
        self.base_tri_ka = 1e4
        self.base_tri_kd = 1.5e-6
        self.base_bending_ke = 5.0
        self.base_bending_kd = 1e-2

        self.scene = ModelBuilder(gravity=-981.0)
        self.viewer = viewer

        self.world_offsets = self._build_world_offsets()

        if self.add_robot:
            self._add_robot_worlds()

        self.table_shape_indices = []
        self._add_tables()

        if self.add_cloth:
            self._add_panels()

        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # hide table primitive auto-rendering and draw them manually after scale conversion
        flags = self.model.shape_flags.numpy()
        for idx in self.table_shape_indices:
            flags[idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # meter-scale table visualization
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(p[0]) * self.viz_scale,
                        float(p[1]) * self.viz_scale,
                        float(p[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
                for p in self.table_positions
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (40.0 * self.viz_scale, 40.0 * self.viz_scale, 10.0 * self.viz_scale)
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5) for _ in range(self.num_envs)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.soft_contact_ke,
            )

        self.viewer.set_model(self.model)
        # Important: this example already places each env at a unique physical world-space
        # offset when building robots / tables / cloth. The viewer also supports its own
        # multi-world layout offsets, and set_model() auto-enables them when shape_world is
        # populated. For this example that causes the visual Franka meshes to be translated
        # a second time, so only one may remain in view. Disable viewer-side world offsets
        # and keep the original physical placement only.
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # wider camera for 5x5 layout
        center_x = 0.5 * (self.world_offsets[:, 0].min() + self.world_offsets[:, 0].max()) * self.viz_scale
        center_y = 0.5 * (self.world_offsets[:, 1].min() + self.world_offsets[:, 1].max()) * self.viz_scale
        self.viewer.set_camera(wp.vec3(center_x - 2.8, center_y + 1.8, 4.5), -20.0, -38.0)

        self.viz_state = self.model.state()

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if self.add_cloth:
            self.capture()

    def _build_world_offsets(self):
        offsets = []
        x0 = -0.5 * (self.grid_cols - 1) * self.cell_spacing_x
        y0 = -0.5 * (self.grid_rows - 1) * self.cell_spacing_y
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                offsets.append((x0 + c * self.cell_spacing_x, y0 + r * self.cell_spacing_y, 0.0))
        return np.array(offsets, dtype=np.float32)

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
            self.table_shape_indices.append(shape_idx)

    def _make_panel_mesh(self, env_id: int):
        rng = np.random.default_rng(1000 + env_id)

        shape_mode = env_id % 3
        width = float(rng.uniform(20.0, 38.0))
        height = float(rng.uniform(20.0, 38.0))
        nx = int(rng.integers(18, 32))
        ny = int(rng.integers(18, 32))

        if shape_mode == 0:  # rectangle
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-width * 0.5,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ width * 0.5,  height * 0.5, 0.0], dtype=np.float32)
        elif shape_mode == 1:  # trapezoid
            top_scale = float(rng.uniform(0.55, 0.9))
            top_w = width * top_scale
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-top_w * 0.5,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ top_w * 0.5,  height * 0.5, 0.0], dtype=np.float32)
        else:  # skewed quad
            skew = float(rng.uniform(-0.25 * width, 0.25 * width))
            c00 = np.array([-width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c10 = np.array([ width * 0.5, -height * 0.5, 0.0], dtype=np.float32)
            c01 = np.array([-width * 0.5 + skew,  height * 0.5, 0.0], dtype=np.float32)
            c11 = np.array([ width * 0.5 + skew,  height * 0.5, 0.0], dtype=np.float32)

        ox, oy, _ = self.world_offsets[env_id]
        target_center = np.array([ox + 20.0, oy - 50.0, 31.5], dtype=np.float32)

        verts = []
        for j in range(ny):
            v = j / (ny - 1)
            for i in range(nx):
                u = i / (nx - 1)
                p = (
                    (1.0 - u) * (1.0 - v) * c00
                    + u * (1.0 - v) * c10
                    + (1.0 - u) * v * c01
                    + u * v * c11
                )
                p[2] += 0.25 * math.sin(u * math.pi) * math.sin(v * math.pi)
                p += target_center
                verts.append(wp.vec3(float(p[0]), float(p[1]), float(p[2])))

        indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                indices.extend([a, b, d, a, d, c])

        tri_ke = self.base_tri_ke * float(rng.uniform(0.35, 2.2))
        tri_ka = self.base_tri_ka * float(rng.uniform(0.35, 2.2))
        edge_ke = self.base_bending_ke * float(rng.uniform(0.25, 3.5))
        density = float(rng.uniform(0.012, 0.05))
        particle_radius = float(rng.uniform(0.65, 1.0))

        return verts, indices, tri_ke, tri_ka, edge_ke, density, particle_radius

    def _add_panels(self):
        for env_id in range(self.num_envs):
            verts, indices, tri_ke, tri_ka, edge_ke, density, particle_radius = self._make_panel_mesh(env_id)
            self.scene.add_cloth_mesh(
                vertices=verts,
                indices=indices,
                rot=wp.quat_identity(),
                pos=wp.vec3(0.0, 0.0, 0.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=density,
                scale=1.0,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                tri_kd=self.base_tri_kd,
                edge_ke=edge_ke,
                edge_kd=self.base_bending_kd,
                particle_radius=particle_radius,
            )

    def set_up_control(self):
        self.control = self.model.control()

        out_dim = 6
        in_dim = self.dof_qd_per_world

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_qd: wp.array[wp.spatial_vector], body_out: wp.array[float]):
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_local_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(self.num_envs, dtype=wp.spatial_vector)

        self.initial_pose = self.model.joint_q.numpy().copy()
        self.initial_pose_world0 = self.initial_pose[: self.dof_q_per_world].copy()

        self.target_xforms = wp.empty(self.num_envs, dtype=wp.transform)
        self.target_xforms_np = np.zeros((self.num_envs, 7), dtype=np.float32)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

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

        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                [2.5, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 26.0, -60.0, 26.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 12.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 15.0, -33.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 15.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -18.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 30.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.0, -20.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets_local = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def compute_body_jacobian_world0(self, model: Model, joint_q: wp.array, joint_qd: wp.array):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = self.dof_qd_per_world
        out_dim = 6

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out])

        J_np = self.J_flat.numpy()
        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            J_np[i * in_dim : (i + 1) * in_dim] = joint_qd.grad.numpy()[:in_dim]
            tape.zero()
        self.J_flat = wp.array(J_np, dtype=float, device=self.model.device)

    def generate_control_joint_qd(self, state_in: State):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        target_local = self.targets_local[current_interval]

        for env_id in range(self.num_envs):
            ox, oy, _ = self.world_offsets[env_id]
            self.target_xforms_np[env_id, 0] = target_local[0] + ox
            self.target_xforms_np[env_id, 1] = target_local[1] + oy
            self.target_xforms_np[env_id, 2] = target_local[2]
            self.target_xforms_np[env_id, 3:] = target_local[3:7]
        self.target_xforms = wp.array(self.target_xforms_np, dtype=wp.transform, device=self.model.device)

        wp.launch(
            compute_ee_delta_batched,
            dim=self.num_envs,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_local_id,
                self.bodies_per_world,
                self.target_xforms,
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian_world0(self.model, state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(6, self.dof_qd_per_world)
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q_all = state_in.joint_q.numpy()
        delta_target_all = self.ee_delta.numpy()
        target_joint_qd_np = np.zeros_like(state_in.joint_qd.numpy())

        for env_id in range(self.num_envs):
            q_start = env_id * self.dof_q_per_world
            qd_start = env_id * self.dof_qd_per_world

            q = q_all[q_start : q_start + self.dof_q_per_world]
            delta_target = delta_target_all[env_id]

            q_des = q.copy()
            q_des[1:] = self.initial_pose_world0[1:]

            K_null = 1.0
            delta_q_null = K_null * (q_des - q)
            delta_q = J_inv @ delta_target + N @ delta_q_null

            delta_q[-2] = target_local[-1] * 4.0 - q[-2]
            delta_q[-1] = target_local[-1] * 4.0 - q[-1]

            target_joint_qd_np[qd_start : qd_start + self.dof_qd_per_world] = delta_q

        self.target_joint_qd.assign(target_joint_qd_np)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                self.model.shape_contact_pair_count = 0
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes("/tables", newton.GeoType.BOX, self.table_viz_scale, self.table_viz_xform, self.table_viz_color)
        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
