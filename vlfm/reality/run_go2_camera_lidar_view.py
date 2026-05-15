# Live viewer for the Go2 front RGB camera + utlidar point cloud.
#
# Default cv2 window: front RGB on the left, top-down BEV of the latest lidar
# sweep on the right (body frame, dog centered, x-forward = up). Pass --open3d
# to also pop up an Open3D 3D viewer. Pass --accumulate to switch the lidar
# from per-sweep body-relative to globally accumulated across frames.
#
# Usage:
#     python -m vlfm.reality.robots.run_go2_camera_lidar_view --iface eth0
#     python -m vlfm.reality.robots.run_go2_camera_lidar_view --iface eth0 --open3d --accumulate
#
# Caveats:
#   * No fake-robot mode — this script needs real camera + lidar streams.
#   * Camera intrinsics in go2_calibration.py are still placeholder TODOs;
#     that's fine here because we never project lidar into the RGB image.
#   * lidar_subscriber._parse_pointcloud2 assumes point_step==16 with
#     [x,y,z,intensity] f32. If the BEV looks empty or garbled while the rest
#     of the pipe is healthy, that parser is the first suspect — verifying
#     it on real firmware is one reason this script exists.
#
# This is NOT a unit test.

import argparse
import datetime as _dt
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from vlfm.reality.lidar_subscriber import LidarSubscriber, transform_cloud
from vlfm.reality.robots.go2_calibration import (
    MAX_OBSTACLE_HEIGHT,
    MIN_OBSTACLE_HEIGHT,
    TF_BODY_TO_UTLIDAR,
)
from vlfm.reality.robots.go2_camera import Go2Camera
from vlfm.reality.robots.go2_frame_ids import Go2FrameIds
from vlfm.reality.robots.go2_robot import Go2Robot


WINDOW_NAME = "go2 camera + lidar"
WAIT_FOR_LIDAR_SEC = 2.0
ACCUM_VOXEL_M = 0.05
ACCUM_DOWNSAMPLE_EVERY = 10
ACCUM_MAX_POINTS = 500_000


def _render_bev(
    cloud_xyz: Optional[np.ndarray],
    bev_range: float,
    bev_resolution: float,
    label: str,
) -> np.ndarray:
    """Top-down BEV of an Nx3 cloud. Coordinates are interpreted with x-forward,
    y-left (Unitree body frame), so x maps to screen-up and y maps to screen-left.
    """
    size = int(2 * bev_range / bev_resolution)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for r_m in (1.0, bev_range):
        cv2.circle(img, (size // 2, size // 2), int(r_m / bev_resolution), (60, 60, 60), 1)

    if cloud_xyz is not None and cloud_xyz.size > 0:
        z = cloud_xyz[:, 2]
        keep = (z >= MIN_OBSTACLE_HEIGHT) & (z <= MAX_OBSTACLE_HEIGHT)
        xy = cloud_xyz[keep, :2]
        in_view = (np.abs(xy[:, 0]) < bev_range) & (np.abs(xy[:, 1]) < bev_range)
        xy = xy[in_view]
        if xy.shape[0] > 0:
            rows = (size // 2 - xy[:, 0] / bev_resolution).astype(np.int32)
            cols = (size // 2 - xy[:, 1] / bev_resolution).astype(np.int32)
            np.clip(rows, 0, size - 1, out=rows)
            np.clip(cols, 0, size - 1, out=cols)
            img[rows, cols] = (255, 255, 255)

    cv2.circle(img, (size // 2, size // 2), 4, (0, 0, 255), -1)
    cv2.arrowedLine(
        img,
        (size // 2, size // 2),
        (size // 2, size // 2 - 16),
        (255, 64, 64),
        2,
        tipLength=0.4,
    )
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _world_to_body_xy(cloud_global: np.ndarray, robot_xy: np.ndarray, robot_yaw: float) -> np.ndarray:
    """Project a global Nx3 cloud into the robot's current body XY for BEV display.

    The BEV in --accumulate mode still wants the dog centered facing up, so we
    de-rotate by -yaw and translate by -robot_xy. Z is preserved for the height filter.
    """
    if cloud_global.size == 0:
        return cloud_global
    dxy = cloud_global[:, :2] - robot_xy
    cy, sy = np.cos(-robot_yaw), np.sin(-robot_yaw)
    rot = np.array([[cy, -sy], [sy, cy]])
    body_xy = dxy @ rot.T
    out = np.empty_like(cloud_global)
    out[:, :2] = body_xy
    out[:, 2] = cloud_global[:, 2]
    return out


def _voxel_downsample(xyz: np.ndarray, voxel: float, o3d_module) -> np.ndarray:
    pcd = o3d_module.geometry.PointCloud()
    pcd.points = o3d_module.utility.Vector3dVector(xyz)
    return np.asarray(pcd.voxel_down_sample(voxel).points, dtype=np.float32)


def _open_video_writer(path: str, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    """frame_size = (width, height)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path} (codec=mp4v, size={frame_size})")
    return writer


def _wait_for_lidar(lidar: LidarSubscriber, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if lidar.get_latest() is not None:
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", type=str, default="eth0", help="Network interface for the Go2 SDK.")
    parser.add_argument("--open3d", action="store_true", help="Also open an Open3D 3D viewer.")
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Accumulate lidar in global frame across sweeps instead of showing only the latest sweep.",
    )
    parser.add_argument("--bev-range", type=float, default=5.0, help="BEV half-extent in meters.")
    parser.add_argument("--bev-resolution", type=float, default=0.05, help="BEV resolution in meters/pixel.")
    parser.add_argument("--rate", type=float, default=15.0, help="Target loop rate (Hz).")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show the raw fisheye frame instead of undistorted pinhole. Use this until "
        "FRONT_CAMERA_K / FRONT_CAMERA_D in go2_calibration.py are filled in — the "
        "placeholder intrinsics make the undistortion produce a black/gradient image.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory for recordings saved with the 's' key (default: current dir).",
    )
    args = parser.parse_args()

    print(f"Connecting to Go2 on iface={args.iface} ...")
    robot = Go2Robot(network_interface=args.iface)
    camera = Go2Camera(robot=robot)
    lidar = LidarSubscriber()

    print(f"Waiting up to {WAIT_FOR_LIDAR_SEC:.1f}s for first lidar sweep ...")
    have_lidar = _wait_for_lidar(lidar, WAIT_FOR_LIDAR_SEC)
    if not have_lidar:
        print("  no lidar yet — continuing; BEV will populate when sweeps arrive.")

    o3d = None
    vis = None
    o3d_pcd = None
    if args.open3d:
        import open3d as _o3d  # noqa: I001 — lazy import keeps --help working without open3d at hand

        o3d = _o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="go2 lidar (open3d)")
        o3d_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(o3d_pcd)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    accum_global: Optional[np.ndarray] = None
    frame_idx = 0
    last_rgb_bgr: Optional[np.ndarray] = None
    period = 1.0 / max(args.rate, 1.0)

    writer: Optional[cv2.VideoWriter] = None
    record_path: Optional[str] = None
    record_size: Optional[Tuple[int, int]] = None  # (W, H)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running. 's' = start/stop recording (saved to {out_dir}), 'q'/ESC = quit.")
    try:
        while True:
            tick = time.time()

            try:
                if args.raw:
                    code, data = camera._video.GetImageSample()
                    if code != 0:
                        raise RuntimeError(f"VideoClient.GetImageSample failed: code={code}")
                    last_rgb_bgr = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if last_rgb_bgr is None:
                        raise RuntimeError("Failed to decode JPEG from VideoClient")
                else:
                    rgb = camera.get_rgb()
                    last_rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:  # noqa: BLE001
                if last_rgb_bgr is None:
                    print(f"  camera not ready: {e}")
                # Keep showing the previous frame.

            cloud_lidar = lidar.get_latest()
            if cloud_lidar is None or cloud_lidar.size == 0:
                bev_cloud = None
                cloud_global = None
            else:
                cloud_body = transform_cloud(cloud_lidar, TF_BODY_TO_UTLIDAR)
                if args.accumulate:
                    tf_lidar_to_global = robot.get_transform(Go2FrameIds.UTLIDAR)
                    cloud_global = transform_cloud(cloud_lidar, tf_lidar_to_global).astype(np.float32)
                    accum_global = (
                        cloud_global if accum_global is None else np.concatenate([accum_global, cloud_global], axis=0)
                    )
                    if frame_idx % ACCUM_DOWNSAMPLE_EVERY == 0 and o3d is not None:
                        accum_global = _voxel_downsample(accum_global, ACCUM_VOXEL_M, o3d)
                    if accum_global.shape[0] > ACCUM_MAX_POINTS:
                        accum_global = accum_global[-ACCUM_MAX_POINTS:]
                    robot_xy, robot_yaw = robot.xy_yaw
                    bev_cloud = _world_to_body_xy(accum_global, robot_xy, robot_yaw)
                else:
                    bev_cloud = cloud_body
                    cloud_global = None

            label = "BEV: global accumulated" if args.accumulate else "BEV: latest sweep (body)"
            bev = _render_bev(bev_cloud, args.bev_range, args.bev_resolution, label)

            if last_rgb_bgr is not None:
                target_h = bev.shape[0]
                scale = target_h / last_rgb_bgr.shape[0]
                target_w = int(round(last_rgb_bgr.shape[1] * scale))
                rgb_disp = cv2.resize(last_rgb_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                combined = cv2.hconcat([rgb_disp, bev])
            else:
                placeholder = np.zeros((bev.shape[0], bev.shape[0] * 4 // 3, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "waiting for camera...",
                    (10, placeholder.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                combined = cv2.hconcat([placeholder, bev])

            if writer is not None and record_size is not None:
                frame_for_writer = combined
                if (combined.shape[1], combined.shape[0]) != record_size:
                    frame_for_writer = cv2.resize(combined, record_size, interpolation=cv2.INTER_AREA)
                writer.write(frame_for_writer)
                cv2.circle(combined, (combined.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                cv2.putText(
                    combined,
                    "REC",
                    (combined.shape[1] - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if writer is None:
                    record_size = (combined.shape[1], combined.shape[0])
                    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_path = os.path.join(out_dir, f"go2_view_{stamp}.mp4")
                    try:
                        writer = _open_video_writer(record_path, record_size, args.rate)
                        print(f"  recording -> {record_path}")
                    except Exception as e:  # noqa: BLE001
                        print(f"  failed to start recording: {e}")
                        writer = None
                        record_size = None
                        record_path = None
                else:
                    writer.release()
                    print(f"  saved {record_path}")
                    writer = None
                    record_size = None
                    record_path = None
            if key in (ord("q"), 27):
                break

            if vis is not None and o3d is not None and o3d_pcd is not None:
                if args.accumulate and accum_global is not None:
                    o3d_pcd.points = o3d.utility.Vector3dVector(accum_global)
                elif cloud_lidar is not None and cloud_lidar.size > 0:
                    cloud_body_full = transform_cloud(cloud_lidar, TF_BODY_TO_UTLIDAR)
                    o3d_pcd.points = o3d.utility.Vector3dVector(cloud_body_full)
                vis.update_geometry(o3d_pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()

            frame_idx += 1
            elapsed = time.time() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        if writer is not None:
            writer.release()
            print(f"  saved {record_path}")
        cv2.destroyAllWindows()
        if vis is not None:
            vis.destroy_window()

    return 0


if __name__ == "__main__":
    sys.exit(main())
