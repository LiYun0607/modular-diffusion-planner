# ONNX splitting scripts

These scripts take the monolithic `diffusion_planner.onnx` (downloaded via
Autoware's `setup-dev-env.sh`) and produce the three independently executable
modules used by our modular C++ runtime:

- `context_encoder.onnx` — runs **once** per planning cycle, output cached
- `dit_core.onnx`        — runs N times inside the C++ DPM-Solver++ loop
- (turn-indicator head is extracted as weights into the C++ post-processor)

## Requirements

```bash
pip install onnx onnx-graphsurgeon numpy
```

## Usage

```bash
# Defaults assume the model is at $DIFFUSION_PLANNER_MODEL_DIR
# (falls back to ~/autoware_data/diffusion_planner/v3.0).
export DIFFUSION_PLANNER_MODEL_DIR=~/autoware_data/diffusion_planner/v3.0

python graphsurgeon_split.py
```

To override paths explicitly:

```bash
python graphsurgeon_split.py \
    --onnx        /path/to/diffusion_planner.onnx \
    --backbone    /path/to/diffusion_planner.onnx \
    --output-dir  /path/to/output/
```

Output files are written next to the original monolithic model:

```
diffusion_planner.onnx          (input, untouched)
context_encoder.onnx            (new — 3,417 nodes, ~30 MB)
dit_core.onnx                   (new — 13,607 nodes, ~28 MB)
```

The C++ ROS 2 node (`planner/autoware_diffusion_planner/`) loads the two
split files and ignores `diffusion_planner.onnx`.

## Analysis tool

`analyze_onnx_split.py` inspects the monolithic graph to identify the
encoder→decoder cut point and node statistics. Useful when adapting the
recipe to a new model version. Run with `--help` for options.

## Model versions supported

- v2.0: `/model/encoder/encoder/fusion/...` and `/model/decoder/decoder/dit/...`
- v3.0: `/model/encoder/fusion/...` and `/model/decoder/dit/...`

The script auto-detects the version from the graph structure.
