# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, subtract_frame_transforms, compute_pose_error
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    # des_pos_b = command[:, :3]
    # des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    des_pos_w = command[:, :3]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    # des_pos_b = command[:, :3]
    # des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    des_pos_w = command[:, :3]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    # des_quat_b = command[:, 3:7]
    # des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    des_quat_w = command[:, 3:7]
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

###################################################### PIH ######################################################

def success_reward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    des_pose_w = command[:, 0:7]
    curr_pose_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 0:7]  # type: ignore

    curr_pos_wrthole, curr_quat_wrthole =  subtract_frame_transforms(
        des_pose_w[:, 0:3], des_pose_w[:, 3:7], curr_pose_w[:, 0:3], curr_pose_w[:, 3:7])
    
    des_pos_wrthole = torch.zeros_like(curr_pos_wrthole).cuda()
    des_quat_wrthole = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(curr_pos_wrthole.size(0), -1).cuda()

    distance_xyz, distance_ang = compute_pose_error(
        des_pos_wrthole,
        des_quat_wrthole,
        curr_pos_wrthole,
        curr_quat_wrthole,
    )

    xyz_error = torch.sum(torch.abs(distance_xyz[:, :3]), dim=-1)
    ang_error = torch.sum(torch.abs(distance_ang), dim=-1)


    # print("des_pose_w", des_pose_w)
    # print("curr_pose_w", curr_pose_w)
    # distance_xyz, distance_ang = compute_pose_error(
    #         des_pose_w[:, :3],
    #         des_pose_w[:, 3:],
    #         curr_pose_w[:, :3],
    #         curr_pose_w[:, 3:7],
    #     )
    
    # print("distance_xyz", distance_xyz)
    # print("distance_ori", distance_ang)

    # xyz_error = torch.sum(torch.abs(distance_xyz[:, :3]), dim=-1)
    # error_ang = torch.sum(torch.abs(distance_ang), dim=-1)

    # print("xyz_error", xyz_error)
    # print("error_ang", error_ang)

    is_xyz_success = torch.where(
        xyz_error < 9e-3,
        torch.ones_like(xyz_error),
        torch.zeros_like(xyz_error),
        )
    
    is_ori_aligned = torch.where(
        ang_error < 5e-2,
        torch.ones_like(ang_error),
        torch.zeros_like(ang_error),
        )
    
    # print("is_xyz_success", is_xyz_success)
    # print("is_ori_aligned", is_ori_aligned)
    is_sucess = 1 * is_xyz_success * is_ori_aligned
    success_reward = 1 * is_sucess

    # print("is_sucess", is_sucess)
    # print("success_reward", success_reward)
    # print("   " * 10)

    return success_reward

def xyz_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current xy positions
    des_pos_w = command[:, :3]
    # print("curr_pos_w", des_pos_w)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # print("curr_pos_w", curr_pos_w)
    xyz_pos_error = torch.norm(curr_pos_w[:, :3] - des_pos_w[:, :3], dim=-1)
    # z_pos_error = torch.norm(curr_pos_w[:, 2] - des_pos_w[:, 2], dim=-1)
    # xyz_pos_error = xy_pos_error + 0.9 * z_pos_error

    # print("xyz_pos_error", xyz_pos_error)
    return xyz_pos_error

def xyz_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    
    xyz_pos_error = torch.norm(curr_pos_w[:, :3] - des_pos_w[:, :3], dim=-1)
    # z_pos_error = torch.norm(curr_pos_w[:, 2] - des_pos_w[:, 2], dim=-1)
    # xyz_pos_error = xy_pos_error + 0.9 * z_pos_error
    
    # is_xy_aligned = torch.where(
    #     xy_pos_error < 4.4e-2,
    #     torch.ones_like(xyz_pos_error),
    #     torch.zeros_like(xyz_pos_error),
    #     )
    
    # aligned_reward = 1 * is_xy_aligned
    
    return 1 - torch.tanh(xyz_pos_error / std)

def z_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pose_w = command[:, 0:7]
    curr_pose_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 0:7]  # type: ignore

    curr_pos_wrthole, curr_quat_wrthole =  subtract_frame_transforms(
        des_pose_w[:, 0:3], des_pose_w[:, 3:7], curr_pose_w[:, 0:3], curr_pose_w[:, 3:7])
    
    des_pos_wrthole = torch.zeros_like(curr_pos_wrthole).cuda()
    des_quat_wrthole = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(curr_pos_wrthole.size(0), -1).cuda()

    distance_xyz, distance_ang = compute_pose_error(
        des_pos_wrthole,
        des_quat_wrthole,
        curr_pos_wrthole,
        curr_quat_wrthole,
    )

    xyz_error = torch.sum(torch.abs(distance_xyz[:, :3]), dim=-1)

      # print("des_pose_w", des_pose_w)
    # print("curr_pose_w", curr_pose_w)
    # distance_xyz, distance_ang = compute_pose_error(
    #         des_pose_w[:, :3],
    #         des_pose_w[:, 3:],
    #         curr_pose_w[:, :3],
    #         curr_pose_w[:, 3:7],
    #     )
    # print("distance_xyz", distance_xyz)
    # print("distance_ori", distance_ang)
    # distance_xy = distance_xyz[:, :2]
    # error_xy = torch.sum(torch.abs(distance_xy), dim=-1)
    ang_error = torch.sum(torch.abs(distance_ang), dim=-1)
    error_x = torch.abs(distance_xyz[:, 0])
    error_y = torch.abs(distance_xyz[:, 1])
    # print("error_xyz", error_xy)
    # print("error_ang", error_ang)
    
    is_x_aligned = torch.where(
        error_x < 9e-4, #8e-3
        torch.ones_like(error_x),
        torch.zeros_like(error_x),
        )
    
    is_y_aligned = torch.where(
        error_y < 9e-4, #8e-3
        torch.ones_like(error_y),
        torch.zeros_like(error_y),
        )
    
    is_ori_aligned = torch.where(
        ang_error < 5e-2, #3e-1
        torch.ones_like(ang_error),
        torch.zeros_like(ang_error),
        )

    # print("error_xyz", is_xy_aligned)
    # print("error_ang", is_ori_aligned)

    height_dist = torch.abs(des_pose_w[:, 2] - curr_pose_w[:, 2])
    # print("[reward_z_error]z_command_error_distance_xy", height_dist)
    reward = 1 - torch.tanh(height_dist / std)
    # print(reward)
    # print(reward * is_xy_aligned * is_ori_aligned)

    return reward * is_x_aligned * is_ori_aligned * is_y_aligned

def force_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)