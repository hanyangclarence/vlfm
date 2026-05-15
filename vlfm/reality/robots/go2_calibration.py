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
# All transforms are 4x4 SE(3) in the form parent_T_child: a point expressed in
# the child frame is mapped into the parent frame by `parent_T_child @ p_child`.
# Body frame: x-forward, y-left, z-up (Unitree convention).
# Translation in meters, rotation as a 3x3 in the upper-left block.

def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = [x, y, z]
    return tf


# body_T_front_camera. Approximate values from published Go2 spec:
# front fisheye sits in the head, ~32 cm forward of body origin and ~4 cm up.
# TODO(A): replace with exact values from the Go2 URDF on hardware bring-up.
TF_BODY_TO_FRONT_CAMERA: np.ndarray = _translation(0.32, 0.0, 0.04)

# body_T_utlidar. The L1 LiDAR sits on top of the body, slightly forward of
# the geometric center.
# TODO(A): replace with exact values from the Go2 URDF on hardware bring-up.
TF_BODY_TO_UTLIDAR: np.ndarray = _translation(0.06, 0.0, 0.18)

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
