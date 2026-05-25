# Modular Diffusion Planner — ITSC 2026

> An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving
>
> Li Yun, Simon Thompson, Yidu Zhang, Ehsan Javanmardi, Manabu Tsukada
>
> *IEEE International Conference on Intelligent Transportation Systems (ITSC) 2026*

This repository accompanies our ITSC 2026 paper. We decompose a monolithic
18,398-node ONNX diffusion planner into three independently executable modules
(context encoder, DiT core, turn-indicator head) and reimplement the
**DPM-Solver++** denoising loop in native C++. The result is a drop-in ROS 2
node within the [Autoware](https://autoware.org/) stack that:

- Enables runtime-configurable solver parameters (no model recompilation)
- Caches the context encoder across denoising steps (3.2× latency reduction)
- Exposes per-step intermediate predictions for observability

We validate numerical equivalence with the monolithic ONNX model
(max error < 10⁻⁵) and benchmark DPM-Solver++ (orders 1, 2) and DDIM across
N ∈ {3, 5, 7, 10, 15, 20} denoising steps in closed-loop AWSIM simulation.

---

## Repository layout

```
itsc2026-release/
├── planner/                                       # Two ROS 2 C++ packages
│   ├── autoware_diffusion_planner_onnx_split_cpp/ # **Algorithm 1 reference impl**
│   │                                              #   - DPM-Solver++ (order 1, 2) in C++
│   │                                              #   - Encoder caching, VP schedule
│   │                                              #   - Anytime denoising support
│   └── autoware_diffusion_planner/                # Upstream-Autoware integration fork
│                                                  #   (monolithic ONNX, for comparison)
├── scripts/onnx_split/                # GraphSurgeon scripts to split monolithic ONNX
│   ├── graphsurgeon_split.py          #   produces context_encoder.onnx + dit_core.onnx
│   └── analyze_onnx_split.py          #   inspects graph to find cut point
├── weights/v2.0/                      # Paper-version split ONNX weights (46 MB)
│   ├── context_encoder_v2.onnx        #   3,417 nodes, 18 MB
│   ├── dit_core.onnx                  #   13,607 nodes, 28 MB
│   ├── turn_indicator_calculator.onnx #   5.6 KB
│   ├── SHA256SUMS
│   └── MODEL_INFO.txt
├── patches/                           # Diff against upstream autowarefoundation/autoware_universe
│   └── modular_planner_vs_upstream.patch
├── benchmarks/                        # Offline benchmark + figure scripts
│   ├── latency_benchmark.py           #   T_enc / T_dit / T_solver via ONNX Runtime
│   ├── offline_solver_benchmark.py    #   FDE/ADE for DPM++(1), DPM++(2), DDIM × N
│   ├── analyze_steps.py               #   step-count sensitivity from closed-loop log
│   ├── gen_figures.py                 #   Reproduces all paper figures
│   └── data/                          #   Cached .npz results from our runs
└── figures/                           # Final paper figure PDFs
```

## Quick start

### 1. Build the ROS 2 node

The repository ships **two** ROS 2 packages under `planner/`:

| Package | Role |
|---|---|
| `autoware_diffusion_planner_onnx_split_cpp` | **The Algorithm 1 reference implementation** — split ONNX + native C++ DPM-Solver++ + encoder caching. This is what the paper benchmarks. |
| `autoware_diffusion_planner` | Upstream Tier4 ROS package consuming the monolithic ONNX. Bundled for Autoware-integration reference and as the *monolithic baseline* in our benchmarks. |

Copy both packages into your Autoware workspace under
`src/` (or `src/universe/autoware_universe/planning/` for the upstream one) and build:

```bash
cd ~/autoware_ws
colcon build --packages-select \
    autoware_diffusion_planner_onnx_split_cpp \
    autoware_diffusion_planner
source install/setup.bash
```

Required model artefacts (downloaded via Autoware's `setup-dev-env.sh`):

```
~/autoware_data/diffusion_planner/v3.1/
├── diffusion_planner.onnx              # full monolithic ONNX (compatibility)
├── context_encoder.onnx                # split: encoder
├── dit_core.onnx                       # split: DiT core
└── diffusion_planner.param.json
```

The split ONNX files are produced from the monolithic `diffusion_planner.onnx`
using [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon):

```bash
pip install onnx onnx-graphsurgeon
export DIFFUSION_PLANNER_MODEL_DIR=~/autoware_data/diffusion_planner/v3.0
python scripts/onnx_split/graphsurgeon_split.py
# writes context_encoder.onnx + dit_core.onnx into $DIFFUSION_PLANNER_MODEL_DIR
```

See [`scripts/onnx_split/README.md`](scripts/onnx_split/README.md) for details
and `--help` for all options.

**Pre-split weights for the version used in the paper** are bundled under
[`weights/v2.0/`](weights/v2.0/) so you can run the benchmarks without
redoing the split. Verify after clone:

```bash
cd weights/v2.0 && sha256sum -c SHA256SUMS
```

### 2. Launch in AWSIM

```bash
ros2 launch autoware_launch planning_simulator.launch.xml \
    map_path:=/path/to/your/map \
    vehicle_model:=sample_vehicle \
    sensor_model:=sample_sensor_kit
```

Or launch the modular node directly:

```bash
ros2 launch autoware_diffusion_planner_onnx_split_cpp diffusion_planner_onnx_split.launch.xml
```

Solver and step count are runtime parameters in
`planner/autoware_diffusion_planner_onnx_split_cpp/config/diffusion_planner.param.yaml`:

```yaml
solver_type: "dpmpp2"     # one of: dpmpp1 | dpmpp2 | ddim
num_steps:    10           # 3, 5, 7, 10, 15, 20
cache_encoder: true        # disable to reproduce monolithic baseline
```

### 3. Reproduce the paper figures

```bash
cd benchmarks
pip install -r requirements.txt   # numpy, onnxruntime, matplotlib
python gen_figures.py             # uses cached data/*.npz from our runs
```

To regenerate the cached `.npz` from your own ONNX runs, point the env vars
at your model + frame-dump directory:

```bash
export DIFFUSION_PLANNER_MODEL_DIR=~/autoware_data/diffusion_planner/v2.0
export DIFFUSION_PLANNER_LOG_DIR=/path/to/captured/awsim_frames

python latency_benchmark.py
python offline_solver_benchmark.py
python gen_figures.py
```

`DIFFUSION_PLANNER_LOG_DIR` should contain per-frame `.bin` tensors dumped by
the planner in debug mode (see
`planner/autoware_diffusion_planner/include/autoware/diffusion_planner/inference/`).

## Citation

```bibtex
@inproceedings{li2026modular,
  title     = {An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving},
  author    = {Li, Yun and Thompson, Simon and Zhang, Yidu and Javanmardi, Ehsan and Tsukada, Manabu},
  booktitle = {2026 IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026}
}
```

See [`CITATION.bib`](CITATION.bib).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

The `planner/autoware_diffusion_planner` package is a derivative of the
upstream [Autoware Universe diffusion planner](https://github.com/autowarefoundation/autoware_universe/tree/main/planning/autoware_diffusion_planner)
(© TIER IV, Inc., Apache 2.0). Our modifications are listed in
[`patches/modular_planner_vs_upstream.patch`](patches/modular_planner_vs_upstream.patch).

## Acknowledgements

- Original diffusion planner architecture and weights:
  [Zheng et al., *Diffusion-Based Planning for Autonomous Driving with
  Flexible Guidance*](https://arxiv.org/abs/2501.15564)
- Upstream Autoware integration: TIER IV, Inc.
- Closed-loop simulator: [AWSIM](https://github.com/tier4/AWSIM)
