# VLFM on Unitree Go2

Fork of [bdaiinstitute/vlfm](https://github.com/bdaiinstitute/vlfm) with a real-robot deployment for the Unitree Go2 (single front fisheye + 360° L1 LiDAR, no arm). See [GO2_PORT_PLAN.md](GO2_PORT_PLAN.md) for the design notes.

Paper: [VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation (ICRA 2024)](https://arxiv.org/abs/2312.03275).

## Install

```bash
conda create -n vlfm python=3.9 -y && conda activate vlfm
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2
pip install -e .[reality]
git clone git@github.com:WongKinYiu/yolov7.git
pip install -e ../unitree_sdk2_python   # Go2 SDK
```

Weights (drop into `data/`):
- `mobile_sam.pt` — https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth` — https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt` — https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth` — already in `data/`

## Run the VLM servers (required for any inference)

```bash
./scripts/launch_vlm_servers.sh
```

Starts a tmux session hosting BLIP-2 / GroundingDINO / MobileSAM / SAM via Flask. Kill the session when done.

## Go2 deployment

**End-to-end ObjectNav on the real robot:**

```bash
python -m vlfm.reality.run_go2_objnav_env \
  env.goal="blue rubbish bin" \
  go2.network_interface=eth0
```

**Hardware-free smoke test** (FakeGo2Robot + synthetic LiDAR + black RGB — verifies the full policy loop without a robot):

```bash
python -m vlfm.reality.run_go2_objnav_env --fake env.goal="blue rubbish bin"
```



### Helper scripts

```bash
# Stand up, drive a 1m square, spin 360°, sit down (locomotion sanity check).
python -m vlfm.reality.run_go2_locomotion_test --iface eth0

# Live viewer: front RGB + BEV of the latest LiDAR sweep (or globally accumulated).
python -m vlfm.reality.run_go2_camera_lidar_view --iface eth0
python -m vlfm.reality.run_go2_camera_lidar_view --iface eth0 --raw       # skip undistortion
python -m vlfm.reality.run_go2_camera_lidar_view --iface eth0 --open3d --accumulate
```

### Useful flags

- `--fake` — FakeGo2Robot + FakeLidarSubscriber (no hardware).
- `--skip-stand` — don't call `stand_up`/`stand_down` (use if dog is already standing).
- Hydra overrides: `env.goal=<str>`, `env.max_lin_dist=<m>`, `env.max_ang_dist=<rad>`, `policy.depth_image_shape="[212,240]"`.



## Citation

```bibtex
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
