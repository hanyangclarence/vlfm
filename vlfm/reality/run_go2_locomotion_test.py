# Manual smoke test for Person A's Go2Robot locomotion stack.
# Stands the dog up, drives a 1m square, spins 360 in place, sits back down.
# Doubles as the gain-tuning harness — re-run after editing constants in
# vlfm/reality/robots/go2_robot.py.
#
# Usage:
#     python -m vlfm.reality.run_go2_locomotion_test --iface eth0
#
# This is NOT a unit test. It expects a real Go2 (or set --fake to use FakeGo2Robot).

import argparse
import sys
import time

import numpy as np

from vlfm.reality.robots.go2_robot import FakeGo2Robot, Go2Robot, POS_TOL, YAW_TOL


def assert_close(actual_xy: np.ndarray, actual_yaw: float, target_xy: np.ndarray, target_yaw: float, label: str) -> None:
    pos_err = float(np.linalg.norm(actual_xy - target_xy))
    yaw_err = float(np.arctan2(np.sin(target_yaw - actual_yaw), np.cos(target_yaw - actual_yaw)))
    ok = pos_err < POS_TOL * 2 and abs(yaw_err) < YAW_TOL * 2
    flag = "OK " if ok else "BAD"
    print(f"  [{flag}] {label}: pos_err={pos_err:.3f}m yaw_err={np.rad2deg(yaw_err):+.2f}°")


def drive_square(robot, side: float = 1.0) -> None:
    print(f"\n--- 1m square (side={side}m) ---")
    start_xy, start_yaw = robot.xy_yaw
    legs = [
        (side, 0.0, 0.0, "leg 1: forward"),
        (0.0, 0.0, np.pi / 2, "leg 2: turn left"),
        (side, 0.0, 0.0, "leg 3: forward"),
        (0.0, 0.0, np.pi / 2, "leg 4: turn left"),
        (side, 0.0, 0.0, "leg 5: forward"),
        (0.0, 0.0, np.pi / 2, "leg 6: turn left"),
        (side, 0.0, 0.0, "leg 7: forward"),
        (0.0, 0.0, np.pi / 2, "leg 8: turn left"),
    ]
    for x, y, yaw, label in legs:
        before_xy, before_yaw = robot.xy_yaw
        cy, sy = np.cos(before_yaw), np.sin(before_yaw)
        target_xy = before_xy + np.array([cy * x - sy * y, sy * x + cy * y])
        target_yaw = before_yaw + yaw
        try:
            robot.set_base_position(x, y, yaw, blocking=True, timeout_sec=15.0)
        except TimeoutError as e:
            print(f"  [BAD] {label}: {e}")
            continue
        actual_xy, actual_yaw = robot.xy_yaw
        assert_close(actual_xy, actual_yaw, target_xy, target_yaw, label)

    end_xy, _ = robot.xy_yaw
    closure = float(np.linalg.norm(end_xy - start_xy))
    print(f"  square closure error: {closure:.3f}m")


def spin_in_place(robot) -> None:
    print("\n--- 360° spin in place ---")
    start_xy, start_yaw = robot.xy_yaw
    for i in range(4):
        before_xy, before_yaw = robot.xy_yaw
        try:
            robot.set_base_position(0.0, 0.0, np.pi / 2, blocking=True, timeout_sec=10.0)
        except TimeoutError as e:
            print(f"  [BAD] quarter {i + 1}: {e}")
            continue
        after_xy, after_yaw = robot.xy_yaw
        drift = float(np.linalg.norm(after_xy - before_xy))
        yaw_delta = np.arctan2(np.sin(after_yaw - before_yaw), np.cos(after_yaw - before_yaw))
        flag = "OK " if drift < POS_TOL * 2 and abs(yaw_delta - np.pi / 2) < YAW_TOL * 2 else "BAD"
        print(f"  [{flag}] quarter {i + 1}: drift={drift:.3f}m yaw_delta={np.rad2deg(yaw_delta):+.2f}°")

    end_xy, end_yaw = robot.xy_yaw
    print(f"  spin closure: drift={np.linalg.norm(end_xy - start_xy):.3f}m yaw={np.rad2deg(end_yaw - start_yaw):+.2f}°")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", type=str, default="eth0", help="Network interface for the Go2 SDK.")
    parser.add_argument("--fake", action="store_true", help="Use FakeGo2Robot (no hardware).")
    parser.add_argument("--side", type=float, default=1.0, help="Square side length in meters.")
    parser.add_argument("--skip-stand", action="store_true", help="Skip stand_up/down (assume dog is already standing).")
    args = parser.parse_args()

    if args.fake:
        print("Using FakeGo2Robot (no hardware).")
        robot = FakeGo2Robot()
    else:
        print(f"Connecting to Go2 on iface={args.iface} ...")
        robot = Go2Robot(network_interface=args.iface)

    if not args.fake and not args.skip_stand:
        print("Standing up ...")
        robot.stand_up()
        time.sleep(1.0)

    drive_square(robot, side=args.side)
    spin_in_place(robot)

    if not args.fake and not args.skip_stand:
        print("\nStanding down ...")
        robot.stand_down()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
