# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDJ Cloth Featherstone+VBD
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples SDJ_cloth_Featherstone_VBD
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.math import transform_twist


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    dst[i] = wp.transform(
        wp.vec3(src[i].p[0] * scale, src[i].p[1] * scale, src[i].p[2] * scale),
        src[i].q,
    )


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
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

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        #       self-contact
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6

        self.bending_ke = 5
        self.bending_kd = 1e-2

        self.scene = ModelBuilder(gravity=-981.0)

        self.viewer = viewer

        # 创建 5x5 网格的机械臂和桌子
        grid_size = 5
        spacing = 150.0  # 每个网格单元的间距（cm）
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算位置
                x_pos = (i - grid_size/2 + 0.5) * spacing
                y_pos = (j - grid_size/2 + 0.5) * spacing
                
                # 创建机械臂
                franka = ModelBuilder()
                self.create_articulation(franka, x_pos, y_pos)
                self.scene.add_world(franka)
                
                # 创建桌子
                self.create_table(x_pos, y_pos)

        # 生成 25 个大小、形状、软硬不同的板片
        if self.add_cloth:
            panel_index = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    # 计算位置
                    x_pos = (i - grid_size/2 + 0.5) * spacing
                    y_pos = (j - grid_size/2 + 0.5) * spacing
                    
                    # 生成随机板片
                    self.generate_random_panel(x_pos, y_pos, panel_index)
                    panel_index += 1

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so scaling the box mesh to be flat would be complicated.
        # Instead we'll draw the table manually as a flat plane.
        self.table_plane = self.viewer.create_mesh(
            newton.examples.get_asset("plane.usda"), scale=(self.table_hx_cm * 2 * self.viz_scale, 1, self.table_hz_cm * 2 * self.viz_scale)
        )

        # Set robot contact properties
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
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
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        # gravity in cm/s²
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def create_articulation(self, builder, x_offset, y_offset):
        import os
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (x_offset - 50.0, y_offset - 50.0, -10.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

    def create_table(self, x_offset, y_offset):
        # add a table (cm scale)
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(x_offset, y_offset, -50.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

    def generate_random_panel(self, x_offset, y_offset, panel_index):
        """生成一个随机形状的板片"""
        import random
        
        # 随机大小
        width = random.uniform(20.0, 40.0)  # 20-40cm
        height = random.uniform(20.0, 40.0)  # 20-40cm
        
        # 随机形状
        resolution = random.randint(8, 12)  # 8-12的分辨率
        
        # 生成网格顶点
        vertices = []
        indices = []
        
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                # 生成随机偏移，使板片形状不规则
                x_offset_panel = random.uniform(-0.1, 0.1) * width
                y_offset_panel = random.uniform(-0.1, 0.1) * height
                z_offset_panel = random.uniform(-0.05, 0.05) * min(width, height)
                
                x = (i / resolution - 0.5) * width + x_offset_panel
                y = (j / resolution - 0.5) * height + y_offset_panel
                z = z_offset_panel
                vertices.append(wp.vec3(x, y, z))
        
        # 生成三角形索引
        for i in range(resolution):
            for j in range(resolution):
                # 第一个三角形
                idx0 = i * (resolution + 1) + j
                idx1 = (i + 1) * (resolution + 1) + j
                idx2 = i * (resolution + 1) + (j + 1)
                indices.extend([idx0, idx1, idx2])
                
                # 第二个三角形
                idx0 = (i + 1) * (resolution + 1) + j
                idx1 = (i + 1) * (resolution + 1) + (j + 1)
                idx2 = i * (resolution + 1) + (j + 1)
                indices.extend([idx0, idx1, idx2])
        
        # 随机位置（叠起来）
        x_pos = x_offset
        y_pos = y_offset
        z_pos = 30.0 + (panel_index % 5) * 5.0  # 高度递增
        target_pos = wp.vec3(x_pos, y_pos, z_pos)
        
        # 计算当前网格的边界框
        bbox_min = [float('inf')] * 3
        bbox_max = [float('-inf')] * 3
        
        for v in vertices:
            for j in range(3):
                bbox_min[j] = min(bbox_min[j], v[j])
                bbox_max[j] = max(bbox_max[j], v[j])
        
        center = [(bbox_min[j] + bbox_max[j]) / 2 for j in range(3)]
        
        # 计算从中心到目标位置的偏移
        final_offset = [
            target_pos[0] - center[0],
            target_pos[1] - center[1],
            target_pos[2] - center[2]
        ]
        
        # 调整顶点位置
        adjusted_vertices = []
        for v in vertices:
            adjusted_v = wp.vec3(
                v[0] + final_offset[0],
                v[1] + final_offset[1],
                v[2] + final_offset[2]
            )
            adjusted_vertices.append(adjusted_v)
        
        # 随机物理参数（软硬不同）
        tri_ke = random.uniform(5000.0, 15000.0)  # 5000-15000
        tri_ka = random.uniform(5000.0, 15000.0)  # 5000-15000
        bending_ke = random.uniform(2.0, 10.0)  # 2-10
        density = random.uniform(0.01, 0.03)  # 0.01-0.03
        
        print(f"Panel {panel_index+1}: size ({width:.1f}x{height:.1f}cm), "
              f"position ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}), "
              f"stiffness (ke={tri_ke:.0f}, ka={tri_ka:.0f}, bending={bending_ke:.1f})")
        
        self.scene.add_cloth_mesh(
            vertices=adjusted_vertices,
            indices=indices,
            rot=wp.quat_identity(),
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=density,
            scale=1.0,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

    def set_up_control(self):
        # joint position targets for key pose tracking
        # Note: this is a simple tracking controller and not a full IK solver,
        #       it works for the specific key poses but may not generalize to arbitrary targets
        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # top left
                [2.5, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, 0.0],
                [2, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, 1.0],
                [2, 26.0, -60.0, 26.0, 1, 0.0, 0.0, 0.0, 1.0],
                [2, 12.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [3, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 0.0],
                # bottom left
                [2, 15.0, -33.0, 31.0, 1, 0.0, 0.0, 0.0, 0.0],
                [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, 0.0],
                [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, 1.0],
                [2, 15.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, 1.0],
                [3, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, 0.0],
                # top right
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, 0.0],
                [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, 0.0],
                [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, 1.0],
                [2, -18.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [3, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, 0.0],
                # bottom right
                [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, 0.0],
                [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, 1.0],
                [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [3, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, 1.0],
                [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, 0.0],
                # bottom
                [2, 0.0, -20.0, 30.0, 1, 0.0, 0.0, 0.0, 0.0],
                [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, 0.0],
                [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, 1.0],
                [2, 0.0, -20.0, 35.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, 1.0],
                [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, 0.0],
                [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = 11  # Franka end effector body ID
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                22.0,
            ],
            wp.quat_identity(),
        )

        # Gripper activation is a binary value (0.0 = open, 1.0 = close)
        self.gripper_activation = 0.0
        self.gripper_activation_target = 0.0

        # Initial pose
        self.initial_pose = self.model.joint_q.numpy().copy()

        # jacobian for inverse kinematics
        self.body_out = wp.zeros(3, dtype=wp.float32, device="cuda")
        self.Jacobian_one_hots = []
        for i in range(3):
            J = wp.zeros(3, dtype=wp.float32, device="cuda")
            J[i] = 1.0
            self.Jacobian_one_hots.append(J)

        self.J = wp.zeros((3, self.model.joint_dof_count), dtype=wp.float32, device="cuda")
        self.J_flat = self.J.reshape(3 * self.model.joint_dof_count)

        # pre-allocate temp state for jacobian computation
        self.temp_state_for_jacobian = State(self.model)
        self.temp_state_for_jacobian.reset()

        # End effector delta
        self.ee_delta = wp.zeros(1, dtype=wp.spatial_vector, device="cuda")

    def capture(self):
        self.graph = wp.capture_begin()
        self.compute_body_jacobian(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            include_rotation=True,
        )
        wp.capture_launch(self.graph)

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute Jacobian of end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            newton.eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    @wp.kernel
    def compute_body_out_kernel(body_qd: wp.array, body_out: wp.array):
        # body id 11 is end effector
        body_out[0] = body_qd[11 * 6 + 0]
        body_out[1] = body_qd[11 * 6 + 1]
        body_out[2] = body_qd[11 * 6 + 2]

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        # After the key poses sequence ends, hold position with zero velocity
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        include_rotation = True

        wp.launch(
            self.compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control (finger positions in cm)
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    @wp.kernel
    def compute_ee_delta(
        body_q: wp.array[wp.transform],
        offset: wp.transform,
        body_id: int,
        bodies_per_world: int,
        target: wp.transform,
        output: wp.array[wp.spatial_vector],
    ):
        # Get the current end effector transform
        ee_xform = body_q[body_id]
        ee_xform = transform_twist(ee_xform, offset)

        # Compute the error in the body frame
        target_body = wp.transform_inverse(target) @ ee_xform
        error = wp.transform_get_translation(target_body)

        # Convert to spatial vector
        output[0] = wp.spatial_vector(
            wp.vec3(error[0], error[1], error[2]),
            wp.vec3(0.0, 0.0, 0.0),
        )

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
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters for visualization
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

        # Swap model shape data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        # Render the table box manually at meter scale
        for i in range(5):
            for j in range(5):
                x_pos = (i - 2) * 150.0
                y_pos = (j - 2) * 150.0
                self.viewer.log_shapes(
                    "/table",
                    newton.GeoType.BOX,
                    wp.vec3(self.table_hx_cm * 2 * self.viz_scale, 1, self.table_hz_cm * 2 * self.viz_scale),
                    wp.transform(
                        (x_pos * self.viz_scale, y_pos * self.viz_scale, 0.0),
                        wp.quat_identity(),
                    ),
                    wp.vec3(0.8, 0.8, 0.8),
                )
        self.viewer.end_frame()

        # Restore simulation shape data
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self):
        grid_size = 5
        spacing = 150.0
        max_extent = (grid_size / 2 + 1) * spacing
        
        p_lower = wp.vec3(-max_extent, -max_extent, -10.0)
        p_upper = wp.vec3(max_extent, max_extent, 150.0)
        
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
