# Owned by Person B. Front fisheye -> undistorted RGB + ZoeDepth nav depth.

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base_robot import BaseRobot
from .go2_calibration import (
    FRONT_CAMERA_D,
    FRONT_CAMERA_K,
    FRONT_CAMERA_NATIVE_SHAPE,
    FRONT_CAMERA_PINHOLE_CX,
    FRONT_CAMERA_PINHOLE_CY,
    FRONT_CAMERA_PINHOLE_FX,
    FRONT_CAMERA_PINHOLE_FY,
    FRONT_CAMERA_PINHOLE_SHAPE,
)
from .go2_frame_ids import Go2FrameIds

try:
    from unitree_sdk2py.go2.video.video_client import VideoClient
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False


# Camera id used in VALUE_MAP_CAMS / object_map paths inside ObjectNavEnv.
FRONT_COLOR: str = "front_color"


class Go2Camera:
    """Front fisheye reader + pinhole undistorter."""

    def __init__(self, robot: BaseRobot, video_client: Optional["VideoClient"] = None):
        if not _HAS_SDK and video_client is None:
            raise ImportError("unitree_sdk2py is not installed.")

        self._robot = robot
        if video_client is None:
            self._video = VideoClient()
            self._video.SetTimeout(3.0)
            self._video.Init()
        else:
            self._video = video_client

        # Precompute the fisheye -> pinhole rectification map once.
        h, w = FRONT_CAMERA_PINHOLE_SHAPE
        new_k = np.array(
            [
                [FRONT_CAMERA_PINHOLE_FX, 0.0, FRONT_CAMERA_PINHOLE_CX],
                [0.0, FRONT_CAMERA_PINHOLE_FY, FRONT_CAMERA_PINHOLE_CY],
                [0.0, 0.0, 1.0],
            ]
        )
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            FRONT_CAMERA_K, FRONT_CAMERA_D, np.eye(3), new_k, (w, h), cv2.CV_16SC2
        )
        self._new_k = new_k

    # ------------------------------------------------------------------------

    def get_rgb(self) -> np.ndarray:
        """Returns the latest undistorted pinhole RGB image (HxWx3, uint8)."""
        code, data = self._video.GetImageSample()
        if code != 0:
            raise RuntimeError(f"VideoClient.GetImageSample failed: code={code}")
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode JPEG from VideoClient")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._undistort(rgb)

    def _undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------------

    def get_camera_data(self) -> Dict[str, Dict[str, Any]]:
        """Mirrors BDSWRobot.get_camera_data shape — used by ObjectNavEnv."""
        rgb = self.get_rgb()
        tf = self._robot.get_transform(Go2FrameIds.FRONT_CAMERA)
        return {
            FRONT_COLOR: {
                "image": rgb,
                "fx": FRONT_CAMERA_PINHOLE_FX,
                "fy": FRONT_CAMERA_PINHOLE_FY,
                "tf_camera_to_global": tf,
            }
        }

    # ------------------------------------------------------------------------

    def get_nav_depth(self, depth_estimator: Any, target_shape: tuple) -> np.ndarray:
        """Returns a [0, 1]-normalized depth image at `target_shape` for PointNav.

        Args:
            depth_estimator: object exposing `infer_pil(PIL.Image) -> np.ndarray` (meters).
                In practice this is RealityITMPolicyV2._depth_model (ZoeDepth).
            target_shape: (H, W) the trained policy expects.
        """
        # TODO(B): verify target_shape against data/pointnav_weights.pth.
        # TODO(B): consider LiDAR-projected depth as an alternative (coordinate with C).
        from PIL import Image

        rgb = self.get_rgb()
        depth_m = depth_estimator.infer_pil(Image.fromarray(rgb))  # meters, float32
        h, w = target_shape
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        max_depth = 5.0  # meters; mirrors max_gripper_cam_depth
        return np.clip(depth_m, 0.0, max_depth) / max_depth
