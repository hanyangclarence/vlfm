# Owned by Person C. Subscribes to rt/utlidar/cloud and exposes the latest
# point cloud as an Nx3 numpy array.

import threading
from typing import Optional

import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False


LIDAR_TOPIC = "rt/utlidar/cloud"  # TODO(C): verify on the robot.


class LidarSubscriber:
    """Latest-cloud cache. Thread-safe: callers get a snapshot copy."""

    def __init__(self, topic: str = LIDAR_TOPIC) -> None:
        if not _HAS_SDK:
            raise ImportError("unitree_sdk2py is not installed.")

        self._lock = threading.Lock()
        self._latest_xyz: Optional[np.ndarray] = None
        self._sub = ChannelSubscriber(topic, PointCloud2_)
        self._sub.Init(self._on_cloud, 1)

    def _on_cloud(self, msg: "PointCloud2_") -> None:
        try:
            xyz = _parse_pointcloud2(msg)
        except Exception:  # noqa: BLE001
            # Drop malformed frames — TODO(C): log instead.
            return
        with self._lock:
            self._latest_xyz = xyz

    def get_latest(self) -> Optional[np.ndarray]:
        """Returns the most recent cloud as Nx3 float32 in the LiDAR frame, or None."""
        with self._lock:
            if self._latest_xyz is None:
                return None
            return self._latest_xyz.copy()


def _parse_pointcloud2(msg: "PointCloud2_") -> np.ndarray:
    """Decode the raw uint8 buffer into Nx3 float32 (x, y, z)."""
    # TODO(C): the proper implementation reads `msg.fields` to find x/y/z offsets
    # and dtype; the cheap version below assumes the standard
    # [x:f32, y:f32, z:f32, intensity:f32] layout (point_step == 16).
    point_step = int(msg.point_step)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n = raw.shape[0] // point_step
    raw = raw[: n * point_step].reshape(n, point_step)
    xyz = np.frombuffer(raw[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)
    # Drop NaN / inf points
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


class FakeLidarSubscriber:
    """Returns a fixed synthetic cloud. Useful for FakeGo2Robot end-to-end tests."""

    def __init__(self, points_lidar_frame: Optional[np.ndarray] = None) -> None:
        if points_lidar_frame is None:
            points_lidar_frame = _default_box_cloud()
        self._points = points_lidar_frame.astype(np.float32)

    def get_latest(self) -> Optional[np.ndarray]:
        return self._points.copy()


def _default_box_cloud() -> np.ndarray:
    """A square room (5m x 5m) with walls at ±2.5m, sampled densely."""
    rng = np.linspace(-2.5, 2.5, 200, dtype=np.float32)
    z = np.full_like(rng, 0.8)
    walls = []
    for x_fixed in (-2.5, 2.5):
        walls.append(np.stack([np.full_like(rng, x_fixed), rng, z], axis=1))
    for y_fixed in (-2.5, 2.5):
        walls.append(np.stack([rng, np.full_like(rng, y_fixed), z], axis=1))
    return np.concatenate(walls, axis=0)


def transform_cloud(xyz_local: np.ndarray, tf_local_to_global: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) to an Nx3 cloud."""
    if xyz_local.size == 0:
        return xyz_local
    homog = np.concatenate([xyz_local, np.ones((xyz_local.shape[0], 1), dtype=xyz_local.dtype)], axis=1)
    return (tf_local_to_global @ homog.T).T[:, :3]
