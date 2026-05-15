# End-to-end VLFM ObjectNav on Unitree Go2.
#
# Prereq: `./scripts/launch_vlm_servers.sh` running in the background to host
# BLIP-2 / GroundingDINO / MobileSAM / SAM. (Same as the Spot path.)
#
# Real-robot run:
#     python -m vlfm.reality.run_go2_objnav_env env.goal="office chair" go2.network_interface=eth0
#
# Hardware-free smoke test (FakeGo2Robot + synthetic LiDAR + black RGB):
#     python -m vlfm.reality.run_go2_objnav_env --fake env.goal="office chair"
#
# Notes:
#   * If FRONT_CAMERA_K / FRONT_CAMERA_D in go2_calibration.py are still
#     placeholder values, the RGB will undistort to garbage. Calibrate first.
#   * Vis images land in ./vis/<MM-DD-HH-MM-SS>/.
#   * Pass --skip-stand if the dog is already standing.

import argparse
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vlfm.policy.reality_policies import RealityITMPolicyV2
from vlfm.reality.go2_objectnav_env import Go2ObjectNavEnv

LIDAR_WAIT_SEC = 5.0


def _build_robot_camera_lidar(cfg: DictConfig, fake: bool) -> Any:
    if fake:
        from vlfm.reality.lidar_subscriber import FakeLidarSubscriber
        from vlfm.reality.robots.go2_robot import FakeGo2Robot

        print("Using FakeGo2Robot + FakeLidarSubscriber + black-image camera (no hardware).")

        class _FakeCamera:
            def get_camera_data(self) -> dict:
                from vlfm.reality.robots.go2_camera import FRONT_COLOR

                rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                return {
                    FRONT_COLOR: {
                        "image": rgb,
                        "fx": 500.0,
                        "fy": 500.0,
                        "tf_camera_to_global": np.eye(4),
                    }
                }

            def get_nav_depth(self, depth_estimator: Any, target_shape: tuple) -> np.ndarray:
                return np.full(target_shape, 0.5, dtype=np.float32)

        return FakeGo2Robot(), _FakeCamera(), FakeLidarSubscriber()

    from vlfm.reality.lidar_subscriber import LidarSubscriber
    from vlfm.reality.robots.go2_camera import Go2Camera
    from vlfm.reality.robots.go2_robot import Go2Robot

    print(f"Connecting to Go2 on iface={cfg.go2.network_interface} ...")
    robot = Go2Robot(network_interface=cfg.go2.network_interface)
    camera = Go2Camera(robot=robot)
    lidar = LidarSubscriber()

    print(f"Waiting up to {LIDAR_WAIT_SEC:.1f}s for first lidar sweep ...")
    deadline = time.time() + LIDAR_WAIT_SEC
    while time.time() < deadline and lidar.get_latest() is None:
        time.sleep(0.05)
    if lidar.get_latest() is None:
        print("  WARNING: no lidar yet — obstacle map will start empty.")

    return robot, camera, lidar


def _run_episode(env: Go2ObjectNavEnv, policy: RealityITMPolicyV2, goal: str) -> None:
    observations = env.reset(goal)
    done = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    st = time.time()
    action = policy.get_action(observations, mask)
    print(f"step 0: get_action took {time.time() - st:.2f}s")
    step_idx = 1
    while not done:
        observations, _, done, _ = env.step(action)
        st = time.time()
        action = policy.get_action(observations, mask, deterministic=True)
        print(f"step {step_idx}: get_action took {time.time() - st:.2f}s")
        mask = torch.ones_like(mask)
        step_idx += 1
    print("Episode finished.")


@hydra.main(version_base=None, config_path="../../config/", config_name="experiments/go2_reality")
def main(cfg: DictConfig) -> None:
    # Parse --fake / --skip-stand off sys.argv since Hydra owns the rest.
    fake = "--fake" in sys.argv
    skip_stand = "--skip-stand" in sys.argv

    print(OmegaConf.to_yaml(cfg))

    policy = RealityITMPolicyV2.from_config(cfg)
    robot, camera, lidar = _build_robot_camera_lidar(cfg, fake=fake)

    if not fake and not skip_stand:
        print("Standing up ...")
        robot.stand_up()
        time.sleep(1.0)

    env = Go2ObjectNavEnv(
        robot=robot,
        camera=camera,
        lidar=lidar,
        depth_model=policy._depth_model,
        nav_depth_shape=tuple(cfg.policy.depth_image_shape),
        max_body_cam_depth=cfg.env.max_body_cam_depth,
        max_gripper_cam_depth=cfg.env.max_gripper_cam_depth,
        max_lin_dist=cfg.env.max_lin_dist,
        max_ang_dist=cfg.env.max_ang_dist,
        time_step=cfg.env.time_step,
    )

    try:
        _run_episode(env, policy, cfg.env.goal)
    finally:
        if not fake:
            print("Stopping robot ...")
            robot.stop()
            if not skip_stand:
                robot.stand_down()


def _strip_argparse_flags() -> None:
    """Drop our argparse-only flags so Hydra doesn't choke on them."""
    for flag in ("--fake", "--skip-stand"):
        while flag in sys.argv:
            sys.argv.remove(flag)


if __name__ == "__main__":
    # Pre-parse to print --help without invoking Hydra
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fake", action="store_true", help="Use FakeGo2Robot + FakeLidar (no hardware).")
    parser.add_argument("--skip-stand", action="store_true", help="Don't call stand_up/stand_down.")
    parser.parse_known_args()
    _strip_argparse_flags()
    main()
