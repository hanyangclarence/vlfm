# ObjectNav environment for Unitree Go2. Replaces ObjectNavEnv's Spot-specific
# step (arm sweep → body rotation) and _get_camera_obs (single front fisheye +
# 360° LiDAR cloud) without touching the policy stack or the Spot path.

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from vlfm.reality.lidar_subscriber import transform_cloud
from vlfm.reality.objectnav_env import ObjectNavEnv
from vlfm.reality.pointnav_env import PointNavEnv
from vlfm.reality.robots.go2_camera import FRONT_COLOR, Go2Camera
from vlfm.reality.robots.go2_frame_ids import Go2FrameIds
from vlfm.utils.geometry_utils import get_fov, pt_from_rho_theta, wrap_heading
from vlfm.utils.img_utils import reorient_rescale_map, resize_images


class Go2ObjectNavEnv(ObjectNavEnv):
    """ObjectNav on Go2: front fisheye RGB for value/object maps, LiDAR cloud for
    obstacle map, body rotation for the initial yaw sweep.
    """

    def __init__(
        self,
        robot: Any,
        camera: Go2Camera,
        lidar: Any,  # LidarSubscriber or FakeLidarSubscriber
        depth_model: Any,  # ZoeDepth from RealityITMPolicyV2._depth_model
        nav_depth_shape: Tuple[int, int],
        max_body_cam_depth: float = 3.5,
        max_gripper_cam_depth: float = 5.0,
        max_lin_dist: float = 0.25,
        max_ang_dist: float = float(np.deg2rad(30)),
        time_step: float = 0.5,
    ) -> None:
        # Skip ObjectNavEnv.__init__ (it expects Spot kwargs); bring up
        # PointNavEnv directly and replicate the small bits ObjectNav adds.
        PointNavEnv.__init__(
            self,
            robot=robot,
            max_body_cam_depth=max_body_cam_depth,
            max_lin_dist=max_lin_dist,
            max_ang_dist=max_ang_dist,
            time_step=time_step,
        )
        self._max_gripper_cam_depth = max_gripper_cam_depth
        self.camera = camera
        self.lidar = lidar
        self.depth_model = depth_model
        self.nav_depth_shape = nav_depth_shape
        date_string = datetime.now().strftime("%m-%d-%H-%M-%S")
        self._vis_dir = date_string
        os.makedirs(f"vis/{self._vis_dir}", exist_ok=True)

    # ------------------------------------------------------------------------
    # reset() is inherited from ObjectNavEnv — sets target_object,
    # tf_episodic_to_global, episodic_start_yaw, then returns self._get_obs().
    # ------------------------------------------------------------------------

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        self._save_visualizations(action)

        if action["arm_yaw"] == -1:
            return self._step_base(action)

        # Initial-sweep yaw: arm_yaw is the absolute target in episodic frame.
        target_episodic_yaw = float(action["arm_yaw"])
        current_episodic_yaw = self._get_compass()
        delta = float(wrap_heading(target_episodic_yaw - current_episodic_yaw))
        try:
            self.robot.set_base_position(0.0, 0.0, delta, blocking=True, timeout_sec=8.0)
        except TimeoutError as e:
            print(f"  [warn] body rotation timed out: {e}")
            self.robot.command_base_velocity(0.0, 0.0)
        time.sleep(0.2)
        self._num_steps += 1
        return self._get_obs(), 0.0, False, {}

    def _step_base(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        ang_dist, lin_dist = self._compute_displacements(action)
        done = action["linear"] == 0.0 and action["angular"] == 0.0

        if "rho_theta" in action:
            rho, theta = action["rho_theta"]
            x_pos, y_pos = pt_from_rho_theta(rho, theta)
            yaw = theta
        else:
            x_pos, y_pos, yaw = lin_dist, 0.0, ang_dist

        if done:
            self.robot.command_base_velocity(0.0, 0.0)
        else:
            try:
                self.robot.set_base_position(
                    float(x_pos), float(y_pos), float(yaw), blocking=True, timeout_sec=5.0
                )
            except TimeoutError as e:
                print(f"  [warn] base move timed out: {e}")
                self.robot.command_base_velocity(0.0, 0.0)

        self._num_steps += 1
        return self._get_obs(), 0.0, done, {}

    # ------------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, Any]:
        robot_xy, robot_heading = self._get_gps(), self._get_compass()
        nav_depth, lidar_cloud_episodic, value_map_rgbd, object_map_rgbd = self._collect_camera_obs()
        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.target_object,
            "obstacle_map_depths": [],
            "lidar_cloud": lidar_cloud_episodic,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }

    def _collect_camera_obs(self) -> Tuple[np.ndarray, np.ndarray, List, List]:
        cam = self.camera.get_camera_data()[FRONT_COLOR]
        rgb = cam["image"]
        fx, fy = cam["fx"], cam["fy"]
        tf_camera_episodic = self.tf_global_to_episodic @ cam["tf_camera_to_global"]

        nav_depth = self.camera.get_nav_depth(self.depth_model, self.nav_depth_shape)

        min_depth = 0.0
        max_depth = self._max_gripper_cam_depth
        fov = get_fov(fx, rgb.shape[1])

        # Depth placeholders — ZoeDepth fills these inside the policy's
        # value/object map updates (`hand_depth_estimated` path in
        # RealityMixin._infer_depth).
        depth_placeholder = np.ones(rgb.shape[:2], dtype=np.float32)
        value_map_rgbd = [(rgb, depth_placeholder, tf_camera_episodic, min_depth, max_depth, fov)]
        object_map_rgbd = [(rgb, depth_placeholder, tf_camera_episodic, min_depth, max_depth, fx, fy)]

        cloud_lidar = self.lidar.get_latest()
        if cloud_lidar is None or cloud_lidar.size == 0:
            cloud_episodic = np.empty((0, 3), dtype=np.float32)
        else:
            tf_lidar_episodic = self.tf_global_to_episodic @ self.robot.get_transform(Go2FrameIds.UTLIDAR)
            cloud_episodic = transform_cloud(cloud_lidar, tf_lidar_episodic).astype(np.float32)

        return nav_depth, cloud_episodic, value_map_rgbd, object_map_rgbd

    # ------------------------------------------------------------------------

    def _save_visualizations(self, action: Dict[str, Any]) -> None:
        info = action.get("info") or {}
        wanted = [k for k in ("annotated_rgb", "annotated_depth", "obstacle_map", "value_map") if k in info]
        if not wanted:
            return
        vis_imgs = []
        time_id = time.time()
        for k in wanted:
            img = cv2.cvtColor(info[k], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"vis/{self._vis_dir}/{time_id}_{k}.png", img)
            if "map" in k:
                img = reorient_rescale_map(img)
            if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
                text = "Target not currently detected"
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                cv2.putText(
                    img,
                    text,
                    (img.shape[1] // 2 - ts[0] // 2, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )
            vis_imgs.append(img)
        vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
        cv2.imwrite(f"vis/{self._vis_dir}/{time_id}.jpg", vis_img)
        if os.environ.get("ZSOS_DISPLAY", "0") == "1":
            cv2.imshow("Visualization", cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(1)
