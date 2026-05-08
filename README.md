# Modular Diffusion Planner — ITSC 2026

> An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving
>
> Li Yun, Simon Thompson, Yidu Zhang, Ehsan Javanmardi, Manabu Tsukada
>
> *IEEE International Conference on Intelligent Transportation Systems (ITSC) 2026*
> [Paper PDF](docs/itsc2026_paper.pdf)

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
├── planner/                           # ROS 2 C++ package (modular planner)
│   └── autoware_diffusion_planner/    #   - inference/ : split-ONNX + DPM-Solver++ in C++
│                                      #   - postprocessing/turn_indicator_manager
├── patches/                           # Diff against upstream autowarefoundation/autoware_universe
│   └── modular_planner_vs_upstream.patch
├── benchmarks/                        # Offline benchmark + figure scripts
│   ├── latency_benchmark.py           #   T_enc / T_dit / T_solver via ONNX Runtime
│   ├── offline_solver_benchmark.py    #   FDE/ADE for DPM++(1), DPM++(2), DDIM × N
│   ├── analyze_steps.py               #   step-count sensitivity from closed-loop log
│   ├── gen_figures.py                 #   Reproduces all paper figures
│   └── data/                          #   Cached .npz results from our runs
├── figures/                           # Final paper PDFs
└── docs/itsc2026_paper.pdf            # Camera-ready paper
```

## Quick start

### 1. Build the ROS 2 node

The package lives at `planner/autoware_diffusion_planner` and depends on
the Autoware Universe stack. Drop it into your Autoware workspace under
`src/universe/autoware_universe/planning/` and build:

```bash
cd ~/autoware_ws
colcon build --packages-select autoware_diffusion_planner
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

The split ONNX files are produced from the monolithic graph using
[ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon);
see [`docs/graphsurgeon_recipe.md`](docs/graphsurgeon_recipe.md) (TODO).

### 2. Launch in AWSIM

```bash
ros2 launch autoware_launch planning_simulator.launch.xml \
    map_path:=/path/to/your/map \
    vehicle_model:=sample_vehicle \
    sensor_model:=sample_sensor_kit
```

Solver and step count are runtime parameters in
`planner/autoware_diffusion_planner/config/diffusion_planner.param.yaml`:

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
