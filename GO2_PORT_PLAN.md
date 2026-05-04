# VLFM → Unitree Go2 Port Plan

A 3-person work split for porting VLFM's real-world deployment from Boston Dynamics Spot to the Unitree Go2.

## Context

VLFM ships with a real-robot stack under `vlfm/reality/` targeting Spot via `bosdyn-client`. The policy stack (`vlfm/policy/`, `vlfm/mapping/`, `vlfm/vlm/`) is robot-agnostic and can stay untouched. The work is to swap out the robot abstraction and adapt to Go2's sensor topology (single front fisheye RGB + 360° L1 LiDAR, no arm, no depth camera).

Reference SDK: `unitree_sdk2_python` (Cyclone DDS, `SportClient`, `VideoClient`, `SportModeState_`, `PointCloud2_`).

---

## Shared Contract (write together, day 1)

Before splitting, agree on these data structures so each person can mock the others' work. Land these in `vlfm/reality/robots/go2_interface.py` and `vlfm/reality/robots/go2_calibration.py`.

### Pose interface

```python
class Go2Pose:
    def xy_yaw() -> tuple[np.ndarray, float]            # (xy[2], yaw) in episodic frame
    def get_transform(frame: str) -> np.ndarray         # 4x4, frame -> global
    # supported frames at minimum: "body", "front_camera", "utlidar"
```

### Observation payload (returned by `ObjectNavEnv._get_obs`, unchanged from Spot)

```python
{
  "nav_depth":           np.ndarray[H, W],          # normalized [0, 1], shape must match trained PointNav policy
  "robot_xy":            np.ndarray[2],
  "robot_heading":       float,
  "objectgoal":          str,
  "obstacle_map_depths": list[(depth, tf, min_d, max_d, fx, fy, fov)],
  "value_map_rgbd":      list[(rgb, depth, tf, min_d, max_d, fov)],
  "object_map_rgbd":     list[(rgb, depth, tf, min_d, max_d, fx, fy)],
}
```

### Calibration constants (one shared module everyone imports)

- Front fisheye: `K`, `D`, image size, undistorted pinhole `(fx, fy, cx, cy)`.
- `tf_body_to_front_camera` (4×4).
- `tf_body_to_utlidar` (4×4).

Source from Unitree URDF / spec where possible; calibrate empirically otherwise.

---

## Person A — Locomotion + State (`Go2Robot`)

### Owns

- `vlfm/reality/robots/go2_robot.py`
- `vlfm/reality/robots/go2_calibration.py` (shared, but A authors it)
- `vlfm/reality/robots/frame_ids.py` (Go2 version)

### Deliverables

1. **SDK init plumbing.** `ChannelFactoryInitialize(0, iface)`, `SportClient`, `MotionSwitcherClient`. `power_on()` analogue is just `StandUp()` then `BalanceStand()`.
2. **Pose subscriber thread.** Subscribe to `rt/lf/sportmodestate` (or `rt/sportmodestate`) typed as `SportModeState_`. Maintain latest pose; expose `xy_yaw`, body velocity, `imu_state.rpy`.
3. **Velocity command.** `command_base_velocity(ang_vel, lin_vel)` → `SportClient.Move(lin_vel, 0.0, ang_vel)`. Auto-stop on near-zero inputs (mirror Spot's behavior at `bdsw_robot.py:55`).
4. **`set_base_position(x, y, yaw, blocking, timeout)` — the most critical function.** This replaces `spot.set_base_position` which VLFM calls every step (`pointnav_env.py:83`). Implementation:
   - Read pose from SportModeState.
   - Compute body-frame error `(Δx, Δy, Δyaw)`.
   - Run a P controller producing `(vx, vy, vyaw)`.
   - Send `SportClient.Move(vx, vy, vyaw)`.
   - Exit when within tolerance, or on timeout. `StopMove()` on exit.
   - Add this to the `BaseRobot` interface (it's not there today; Spot calls it via `self.robot.spot.set_base_position`).
5. **`get_transform(frame)`.** Compose body pose (from SportModeState) with hard-coded body→frame offsets from `go2_calibration`.
6. **Stub the arm/gripper.** `arm_joints`, `set_arm_joints`, `open_gripper` as no-ops. Keep them on the class so `ObjectNavEnv` doesn't crash if it still calls them.
7. **`FakeGo2Robot`.** Mirror Spot's `FakeRobot` (`base_robot.py:83`) — return synthetic pose so B and C can develop without hardware.

### Test plan

- Drive a 1m × 1m square in an empty area.
- Verify `set_base_position` reaches goals within ≈10 cm and ≈5°.
- Spin in place to test angular control independently.

### Blocks

Nobody. Can start immediately.

---

## Person B — Camera & Nav Depth

### Owns

- `vlfm/reality/robots/go2_camera.py`
- Modifications to `vlfm/policy/reality_policies.py` (depth inference)
- Modifications to `vlfm/reality/pointnav_env.py::_get_nav_depth`

### Deliverables

1. **`VideoClient` wrapper.** Returns the front fisheye as a numpy RGB image at known shape. Handle reconnects, JPEG decode (`cv2.imdecode`).
2. **Fisheye calibration & undistortion.** Calibrate intrinsics with a chessboard (or use Unitree's published values). Provide `undistort_to_pinhole(img) -> (img, fx, fy)`. Critical: Spot's hand camera is pinhole; Go2's front is fisheye. The pinhole math throughout `vlfm/mapping/` will be wrong on raw fisheye.
3. **Nav-depth pipeline.** Two paths — start with #1, add #2 if quality is poor:
   - **ZoeDepth on undistorted RGB.** `RealityMixin._infer_depth` (`reality_policies.py:156`) already loads `ZoeD_NK` — reuse it. Reshape to whatever input shape the trained PointNav checkpoint expects (verify against `data/pointnav_weights.pth`). Skip the `uint16/1000` normalization in `_norm_depth` (`pointnav_env.py:143`) since Zoe returns float meters.
   - **LiDAR-to-depth.** Project `rt/utlidar/cloud` into the front-camera frustum (need `tf_camera_to_utlidar` from A's calibration), then densify. Coordinate with C who already has the LiDAR subscriber.
4. **Value-map / object-map RGB-D entries.** Implement `get_camera_data` returning `{image, fx, fy, tf_camera_to_global}` per `BDSWRobot.get_camera_data` (`bdsw_robot.py:90`). Object-map depth can stay as the all-ones placeholder Spot uses (`objectnav_env.py:170`) — the policy fills it via ZoeDepth.
5. **Edit `ObjectNavEnv` constants** at `objectnav_env.py:21-37`:
   - `VALUE_MAP_CAMS = ["front_color"]`.
   - `POINT_CLOUD_CAMS = []` — C is replacing this with LiDAR.
   - Drop the body-cam branches in `_get_camera_obs`.

### Test plan

- Stream front camera, undistort, overlay ZoeDepth, verify reasonable scale by measuring known objects (e.g. a chair at 2m should read ≈2m).
- Save a side-by-side: raw fisheye, undistorted pinhole, ZoeDepth heatmap.

### Blocks

Needs A's `xy_yaw` and `get_transform("front_camera")`. Use `FakeGo2Robot` until A ships, then swap.

---

## Person C — Obstacle Map from LiDAR + Integration

### Owns

- `vlfm/reality/lidar_subscriber.py`
- Modifications to `vlfm/mapping/obstacle_map.py`
- `vlfm/reality/run_go2_objnav_env.py` (entry point)
- `config/experiments/go2_reality.yaml`
- The env-level rewrite in `vlfm/reality/objectnav_env.py`

### Deliverables

1. **LiDAR subscriber.** `ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)`. Maintain a latest-cloud buffer. Parse `PointCloud2_.data` (uint8 buffer) using `fields[]` (`x, y, z, intensity` as float32). Return `np.ndarray[N, 3]` in the LiDAR frame.
2. **`ObstacleMap.update_from_pointcloud(points_global, agent_xy)`.** Read `vlfm/mapping/obstacle_map.py` first to understand its current cell-update scheme. Write a parallel path that:
   - Filters points by `min_obstacle_height` / `max_obstacle_height` (already configured at `config/experiments/reality.yaml:19-20`).
   - Bins surviving points into the top-down grid as obstacles.
   - Marks cells in a radius around the agent as explored — the LiDAR's free-space coverage replaces depth-camera frustum-marching.
3. **Rewrite the action branch in `ObjectNavEnv.step`** at `objectnav_env.py:102-112`:
   - Today: `action["arm_yaw"] != -1` rotates the gripper camera through bootstrap yaws.
   - Replace with: `robot.set_base_position(0, 0, target_yaw, blocking=True)` — spin the body in place to that yaw. This replaces the 8-yaw arm sweep (`reality_policies.py:16`).
   - Remove the gripper-stow at lines 105-107.
4. **Replace `_get_camera_obs`'s obstacle path.** Instead of iterating `POINT_CLOUD_CAMS`, call C's LiDAR subscriber. Build `obstacle_map_depths` as a single LiDAR-derived entry — or cleaner, change the env to pass the cloud directly and bypass the depth-camera abstraction. Coordinate with how `RealityMixin._cache_observations` consumes it (`reality_policies.py:113-138`).
5. **Entry point — `run_go2_objnav_env.py`.** Mirror `run_bdsw_objnav_env.py:1`. Pulls A's `Go2Robot`, B's camera, C's LiDAR; instantiates `ObjectNavEnv` and `RealityITMPolicyV2`; runs the policy loop.
6. **Hydra config — `config/experiments/go2_reality.yaml`.** Clone from `reality.yaml` with Go2-tuned `agent_radius`, `max_lin_dist`, `max_ang_dist`, depth ranges.

### Test plan

- Stand the dog still, verify obstacle map populates from LiDAR around the room.
- Teleop a few meters and confirm the map updates correctly with body motion.
- Run the full pipeline with a goal of "office chair" in a controlled space.

### Blocks

For #2 and #4 needs A's `xy_yaw` / `get_transform`. For #5 needs B's camera. Use `FakeGo2Robot` + a recorded RGB folder until those land.

---

## Timeline (≈3 weeks)

| Week | A (Locomotion) | B (Perception) | C (Mapping / Integration) |
|---|---|---|---|
| 1 | SDK init, pose subscriber, `Move`, `command_base_velocity` | VideoClient wrapper, fisheye calibration, undistortion | LiDAR subscriber, parse PointCloud2, visualize cloud |
| 2 | `set_base_position` P-controller, `get_transform`, `FakeGo2Robot` | ZoeDepth nav-depth path, `get_camera_data` | `update_from_pointcloud`, env step rewrite (arm_yaw → body spin) |
| 3 | Hardware tuning, gain tuning, edge cases | Depth-quality eval, fallback handling | End-to-end runner, config, joint debug on robot |

## Integration Milestones

- **End of week 1.** All three can `import` each other's modules with stub data flowing. No robot needed.
- **End of week 2.** Simulated end-to-end run using `FakeGo2Robot` + recorded RGB + recorded LiDAR bag. `ObjectNavEnv.step` returns a valid obs dict.
- **End of week 3.** Dog stands up, scans, navigates to "office chair" in a controlled room.

---

## Risks & Watch-outs

- **Pose drift.** Go2's odom integrates IMU + leg kinematics; over a 5-min episode it can drift meters. The episodic frame in `objectnav_env.reset` is fine for one episode but expect the obstacle map to slowly skew. If drift is bad: add a fallback (T265 / Realsense odom) or shorten episodes.
- **Coordinate frames.** Three frames matter — `body`, `front_camera`, `utlidar`. Get them aligned in week 1; debugging frame errors after the fact eats days. Visualize all three together once and lock it in.
- **Trained PointNav input shape.** Don't change the nav-depth tensor shape without checking `data/pointnav_weights.pth`. A shape mismatch silently produces garbage actions instead of crashing.
- **Fisheye undistortion crop.** Aggressive undistortion crops the FOV — you'll lose edges. Tune the alpha parameter to balance FOV vs. distortion.
- **LiDAR rate vs. control rate.** L1 LiDAR is ~10 Hz. The control loop wants pose at higher rate — keep pose from SportModeState (~50 Hz) and use LiDAR only for the obstacle map.

---

## File Map (what gets created vs. modified)

**New files:**
- `vlfm/reality/robots/go2_robot.py` (A)
- `vlfm/reality/robots/go2_calibration.py` (shared, A authors)
- `vlfm/reality/robots/go2_interface.py` (shared contract)
- `vlfm/reality/robots/go2_camera.py` (B)
- `vlfm/reality/lidar_subscriber.py` (C)
- `vlfm/reality/run_go2_objnav_env.py` (C)
- `config/experiments/go2_reality.yaml` (C)

**Modified files:**
- `vlfm/reality/robots/base_robot.py` — add `set_base_position` to interface (A)
- `vlfm/reality/objectnav_env.py` — camera lists, action-branch rewrite, obstacle path (B + C)
- `vlfm/reality/pointnav_env.py` — replace `self.robot.spot.*` calls with `BaseRobot` methods, swap nav-depth source (B)
- `vlfm/policy/reality_policies.py` — extend `_infer_depth` for nav depth (B)
- `vlfm/mapping/obstacle_map.py` — add `update_from_pointcloud` (C)

**Untouched:** `vlfm/policy/` (except `reality_policies.py`), `vlfm/mapping/value_map.py`, `vlfm/mapping/object_point_cloud_map.py`, `vlfm/vlm/` — the policy stack is robot-agnostic.
