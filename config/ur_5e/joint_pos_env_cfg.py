# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.PiH.PiH_env_cfg import PiHEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG, UR5e_PEG_CFG# isort: skip


##
# Environment configuration
##


@configclass
class UR5ePiHEnvCfg(PiHEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = UR5e_PEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        # override rewards
        self.rewards.ori_error.params["asset_cfg"].body_names = ["peg"]
        # PiH sepecific rewards:
        self.rewards.xyz_error.params["asset_cfg"].body_names = ["peg"]
        self.rewards.xyz_error_tanh.params["asset_cfg"].body_names = ["peg"]
        self.rewards.aligned_insert.params["asset_cfg"].body_names = ["peg"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "peg"
        self.commands.ee_pose.ranges.pitch = (0, 0)


@configclass
class UR5ePiHEnvCfg_PLAY(UR5ePiHEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
