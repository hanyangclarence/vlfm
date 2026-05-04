# Owned by Person C. Entry point for VLFM ObjectNav on Go2.
# Mirrors run_bdsw_objnav_env.py but wires in Go2Robot / Go2Camera / LidarSubscriber.

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from vlfm.policy.reality_policies import RealityConfig, RealityITMPolicyV2
from vlfm.reality.lidar_subscriber import LidarSubscriber
from vlfm.reality.objectnav_env import ObjectNavEnv
from vlfm.reality.robots.go2_camera import Go2Camera
from vlfm.reality.robots.go2_robot import Go2Robot


@hydra.main(version_base=None, config_path="../../config/", config_name="experiments/go2_reality")
def main(cfg: RealityConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    policy = RealityITMPolicyV2.from_config(cfg)
    robot = Go2Robot(network_interface=cfg.go2.network_interface)
    robot.stand_up()
    camera = Go2Camera(robot=robot)
    lidar = LidarSubscriber()

    # TODO(C): ObjectNavEnv currently uses Spot-specific paths in
    # _get_camera_obs and the arm_yaw branch of step(). Subclass it as
    # Go2ObjectNavEnv and override:
    #   - _get_camera_obs: build value_map_rgbd from `camera`,
    #     obstacle_map_depths from `lidar`, object_map_rgbd from `camera`.
    #   - step: when action["arm_yaw"] != -1, call
    #     robot.set_base_position(0, 0, action["arm_yaw"], blocking=True)
    #     instead of moving the arm.
    env = ObjectNavEnv(
        robot=robot,
        max_body_cam_depth=cfg.env.max_body_cam_depth,
        max_gripper_cam_depth=cfg.env.max_gripper_cam_depth,
        max_lin_dist=cfg.env.max_lin_dist,
        max_ang_dist=cfg.env.max_ang_dist,
        time_step=cfg.env.time_step,
    )

    try:
        run_env(env, policy, cfg.env.goal)
    finally:
        robot.stop()


def run_env(env: ObjectNavEnv, policy: RealityITMPolicyV2, goal: str) -> None:
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bool)
    st = time.time()
    action = policy.get_action(observations, mask)
    print(f"get_action took {time.time() - st:.2f}s")
    while not done:
        observations, _, done, _ = env.step(action)
        st = time.time()
        action = policy.get_action(observations, mask, deterministic=True)
        print(f"get_action took {time.time() - st:.2f}s")
        mask = torch.ones_like(mask)
        if done:
            print("Episode finished")
            break


if __name__ == "__main__":
    main()
