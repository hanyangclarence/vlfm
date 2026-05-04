# Owned by Person A. Wraps unitree_sdk2py for the VLFM BaseRobot interface.

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_robot import BaseRobot
from .go2_calibration import (
    TF_BODY_TO_FRONT_CAMERA,
    TF_BODY_TO_UTLIDAR,
    TF_CAMERA_OPTICAL_TO_XYZ,
)
from .go2_frame_ids import Go2FrameIds

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False


SPORT_STATE_TOPIC = "rt/lf/sportmodestate"  # TODO(A): verify against firmware version

# P-controller gains for set_base_position. TODO(A): tune on hardware.
KP_LIN = 0.8
KP_ANG = 1.2
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.0
POS_TOL = 0.10  # meters
YAW_TOL = np.deg2rad(5.0)


class Go2Robot(BaseRobot):
    """BaseRobot implementation backed by unitree_sdk2py (Go2)."""

    def __init__(self, network_interface: str):
        if not _HAS_SDK:
            raise ImportError("unitree_sdk2py is not installed. `pip install -e ../unitree_sdk2_python`.")

        ChannelFactoryInitialize(0, network_interface)

        self._sport = SportClient()
        self._sport.SetTimeout(10.0)
        self._sport.Init()

        self._video = VideoClient()
        self._video.SetTimeout(3.0)
        self._video.Init()

        self._state_lock = threading.Lock()
        self._latest_state: Optional[SportModeState_] = None
        self._state_sub = ChannelSubscriber(SPORT_STATE_TOPIC, SportModeState_)
        self._state_sub.Init(self._on_sport_state, 10)

        self._wait_for_first_state(timeout_sec=5.0)

    # ----- Lifecycle ---------------------------------------------------------

    def stand_up(self) -> None:
        self._sport.StandUp()
        time.sleep(2.0)
        self._sport.BalanceStand()

    def stand_down(self) -> None:
        self._sport.StandDown()

    def stop(self) -> None:
        self._sport.StopMove()

    # ----- Pose / state ------------------------------------------------------

    def _on_sport_state(self, msg: "SportModeState_") -> None:
        with self._state_lock:
            self._latest_state = msg

    def _wait_for_first_state(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._state_lock:
                if self._latest_state is not None:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for {SPORT_STATE_TOPIC}")

    def _state(self) -> "SportModeState_":
        with self._state_lock:
            assert self._latest_state is not None
            return self._latest_state

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        s = self._state()
        x, y = float(s.position[0]), float(s.position[1])
        yaw = float(s.imu_state.rpy[2])
        return np.array([x, y]), yaw

    @property
    def arm_joints(self) -> np.ndarray:
        return np.zeros(6)  # Go2 has no arm

    # ----- Transforms --------------------------------------------------------

    def get_transform(self, frame: str = Go2FrameIds.BODY) -> np.ndarray:
        body_to_global = self._body_to_global()
        if frame == Go2FrameIds.BODY:
            return body_to_global
        if frame == Go2FrameIds.FRONT_CAMERA:
            return body_to_global @ TF_BODY_TO_FRONT_CAMERA @ TF_CAMERA_OPTICAL_TO_XYZ
        if frame == Go2FrameIds.UTLIDAR:
            return body_to_global @ TF_BODY_TO_UTLIDAR
        raise ValueError(f"Unknown frame: {frame}")

    def _body_to_global(self) -> np.ndarray:
        s = self._state()
        roll, pitch, yaw = (float(v) for v in s.imu_state.rpy)
        tf = np.eye(4)
        tf[:3, :3] = _rpy_to_matrix(roll, pitch, yaw)
        tf[:3, 3] = [float(s.position[0]), float(s.position[1]), float(s.position[2])]
        return tf

    # ----- Velocity / position commands -------------------------------------

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        if abs(ang_vel) < 0.01 and abs(lin_vel) < 0.01:
            self._sport.StopMove()
        else:
            self._sport.Move(float(lin_vel), 0.0, float(ang_vel))

    def set_base_position(
        self,
        x_pos: float,
        y_pos: float,
        yaw: float,
        blocking: bool = True,
        timeout_sec: float = 10.0,
    ) -> None:
        # Treat (x_pos, y_pos, yaw) as relative to the current pose (matches the
        # `relative=True` semantics in pointnav_env.py:88).
        start_xy, start_yaw = self.xy_yaw
        cos_y, sin_y = np.cos(start_yaw), np.sin(start_yaw)
        target_xy = start_xy + np.array([cos_y * x_pos - sin_y * y_pos, sin_y * x_pos + cos_y * y_pos])
        target_yaw = _wrap(start_yaw + yaw)

        if not blocking:
            # TODO(A): non-blocking variant — spawn a worker thread that holds
            # the controller until convergence. For now require blocking=True.
            raise NotImplementedError("Non-blocking set_base_position is a TODO")

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            xy, current_yaw = self.xy_yaw
            err = target_xy - xy
            if np.linalg.norm(err) < POS_TOL and abs(_wrap(target_yaw - current_yaw)) < YAW_TOL:
                self._sport.StopMove()
                return

            # Body-frame error
            cy, sy = np.cos(current_yaw), np.sin(current_yaw)
            err_body = np.array([cy * err[0] + sy * err[1], -sy * err[0] + cy * err[1]])
            yaw_err = _wrap(target_yaw - current_yaw)

            vx = float(np.clip(KP_LIN * err_body[0], -MAX_LIN_VEL, MAX_LIN_VEL))
            vy = float(np.clip(KP_LIN * err_body[1], -MAX_LIN_VEL, MAX_LIN_VEL))
            vyaw = float(np.clip(KP_ANG * yaw_err, -MAX_ANG_VEL, MAX_ANG_VEL))
            self._sport.Move(vx, vy, vyaw)
            time.sleep(0.05)

        self._sport.StopMove()
        raise TimeoutError(f"set_base_position did not converge in {timeout_sec}s")

    # ----- Arm / gripper (no-ops for Go2) -----------------------------------

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        return

    def open_gripper(self) -> None:
        return

    # ----- Cameras -----------------------------------------------------------

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        # Forward to Go2Camera. TODO(B): implement and wire in.
        raise NotImplementedError("Use vlfm.reality.robots.go2_camera.Go2Camera")

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        # Same as above — Go2Camera produces (image, fx, fy, tf_camera_to_global).
        raise NotImplementedError("Use vlfm.reality.robots.go2_camera.Go2Camera")


class FakeGo2Robot(BaseRobot):
    """Hardware-free stand-in. Lets B and C develop without the dog."""

    def __init__(self) -> None:
        self._xy = np.zeros(2, dtype=np.float64)
        self._yaw = 0.0

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        return self._xy.copy(), self._yaw

    @property
    def arm_joints(self) -> np.ndarray:
        return np.zeros(6)

    def get_transform(self, frame: str = Go2FrameIds.BODY) -> np.ndarray:
        body_to_global = np.eye(4)
        body_to_global[:3, :3] = _rpy_to_matrix(0.0, 0.0, self._yaw)
        body_to_global[:2, 3] = self._xy
        if frame == Go2FrameIds.BODY:
            return body_to_global
        if frame == Go2FrameIds.FRONT_CAMERA:
            return body_to_global @ TF_BODY_TO_FRONT_CAMERA @ TF_CAMERA_OPTICAL_TO_XYZ
        if frame == Go2FrameIds.UTLIDAR:
            return body_to_global @ TF_BODY_TO_UTLIDAR
        raise ValueError(f"Unknown frame: {frame}")

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        dt = 0.1
        self._yaw = _wrap(self._yaw + ang_vel * dt)
        self._xy += np.array([np.cos(self._yaw), np.sin(self._yaw)]) * lin_vel * dt

    def set_base_position(
        self,
        x_pos: float,
        y_pos: float,
        yaw: float,
        blocking: bool = True,
        timeout_sec: float = 10.0,
    ) -> None:
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        self._xy += np.array([cy * x_pos - sy * y_pos, sy * x_pos + cy * y_pos])
        self._yaw = _wrap(self._yaw + yaw)

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        return

    def open_gripper(self) -> None:
        return

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        return {src: np.zeros((480, 640, 3), dtype=np.uint8) for src in camera_source}

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for src in srcs:
            out[src] = {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "fx": 500.0,
                "fy": 500.0,
                "tf_camera_to_global": self.get_transform(Go2FrameIds.FRONT_CAMERA),
            }
        return out


# ---------- helpers ----------


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx
