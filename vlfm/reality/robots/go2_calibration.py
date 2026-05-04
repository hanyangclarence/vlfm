# Shared Go2 calibration constants. Owned by Person A; consumed by B and C.
# All TODO values must be measured / sourced from Unitree URDF before hardware bring-up.

from typing import Tuple

import numpy as np

# --- Front fisheye camera ---------------------------------------------------
# Native resolution of the JPEG returned by VideoClient.GetImageSample.
# TODO(A): confirm against actual frames from the robot.
FRONT_CAMERA_NATIVE_SHAPE: Tuple[int, int] = (1080, 1920)  # (H, W)

# Fisheye intrinsics (raw, distorted). TODO(B): calibrate with chessboard.
FRONT_CAMERA_K: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],  # TODO: fx, 0, cx
        [0.0, 1.0, 0.0],  # TODO: 0, fy, cy
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FRONT_CAMERA_D: np.ndarray = np.zeros(4, dtype=np.float64)  # TODO(B): k1, k2, k3, k4 (fisheye)

# Pinhole intrinsics produced after undistortion. Set by the calibration
# pipeline in go2_camera.py once the FOV-vs-distortion alpha is chosen.
FRONT_CAMERA_PINHOLE_SHAPE: Tuple[int, int] = (480, 640)  # (H, W). TODO(B): finalize.
FRONT_CAMERA_PINHOLE_FX: float = 1.0  # TODO(B)
FRONT_CAMERA_PINHOLE_FY: float = 1.0  # TODO(B)
FRONT_CAMERA_PINHOLE_CX: float = 320.0  # TODO(B)
FRONT_CAMERA_PINHOLE_CY: float = 240.0  # TODO(B)

# --- Static transforms -------------------------------------------------------
# All transforms are 4x4 SE(3), child_T_parent semantics: tf @ point_in_parent.

def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)

# body -> front_camera. TODO(A): pull from Unitree URDF.
TF_BODY_TO_FRONT_CAMERA: np.ndarray = _identity()

# body -> utlidar. TODO(A): pull from Unitree URDF.
TF_BODY_TO_UTLIDAR: np.ndarray = _identity()

# Camera-convention to xyz-convention rotation, copied from Spot
# (objectnav_env.py:141). Verify sign convention for Go2's front camera.
TF_CAMERA_OPTICAL_TO_XYZ: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# --- Obstacle map filtering --------------------------------------------------
# Heights (meters, in body frame) outside this band are dropped before binning
# the LiDAR cloud into the top-down obstacle grid.
MIN_OBSTACLE_HEIGHT: float = 0.1
MAX_OBSTACLE_HEIGHT: float = 1.5
