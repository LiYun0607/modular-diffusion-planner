#!/usr/bin/env python3
"""
GraphSurgeon-based ONNX Model Splitting

Splits diffusion_planner.onnx into:
1. context_encoder.onnx: Static inputs → context_embedding [226, B, 256]
2. dit_core.onnx: context_embedding + timestep + sampled_trajectories → dit_prediction

Key features:
- Encoder runs ONCE per planning cycle
- DiT runs N times in DPM-Solver loop with dynamic timestep
- Turn indicator computed separately (see extract_turn_indicator.py)

Supports both v2.0 and v3.0 model structures:
- v2.0: /model/encoder/encoder/fusion/... and /model/decoder/decoder/dit/...
- v3.0: /model/encoder/fusion/... and /model/decoder/dit/...
"""

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from pathlib import Path
import argparse


def detect_model_version(model) -> str:
    """Detect if model is v2.0 or v3.0 based on node paths."""
    for node in model.graph.node:
        if '/model/encoder/encoder/fusion/' in node.name:
            return "v2.0"
        if '/model/encoder/fusion/' in node.name:
            return "v3.0"
    return "unknown"


def get_cut_points(version: str) -> dict:
    """Get the correct cut points for the model version."""
    if version == "v2.0":
        return {
            "encoder_output": "/model/encoder/encoder/fusion/norm/LayerNormalization_output_0",
            "dit_output": "/model/decoder/decoder/dit/final_layer/proj/proj.4/Add_output_0",
            "timestep_const": "/model/encoder/encoder/ego_encoder/Constant_7",
        }
    elif version == "v3.0":
        return {
            "encoder_output": "/model/encoder/fusion/norm/LayerNormalization_output_0",
            "dit_output": "/model/decoder/dit/final_layer/proj/proj.4/Add_output_0",
            "timestep_const": "/model/encoder/ego_encoder/Constant_7",
        }
    else:
        raise ValueError(f"Unknown model version: {version}")


def create_context_encoder(model_path: str, output_path: str) -> bool:
    """
    Extract context encoder from diffusion_planner.onnx.

    Output: context_embedding [226, B, 256]
    """
    print("=" * 80)
    print("Creating context_encoder.onnx")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading model: {model_path}")
    model = onnx.load(model_path)

    # Detect version
    version = detect_model_version(model)
    print(f"   Detected model version: {version}")
    cut_points = get_cut_points(version)

    graph = gs.import_onnx(model)

    print(f"   Original nodes: {len(graph.nodes)}")
    print(f"   Original inputs: {len(graph.inputs)}")

    # Find the encoder output tensor
    cut_point = cut_points["encoder_output"]

    print(f"\n2. Finding cut point: {cut_point}")

    encoder_output = None
    for tensor_name, tensor in graph.tensors().items():
        if tensor_name == cut_point:
            encoder_output = tensor
            break

    if encoder_output is None:
        print(f"   ERROR: Cut point not found!")
        return False

    print(f"   Found: {encoder_output.name}")
    print(f"   Shape: {encoder_output.shape}")
    print(f"   Dtype: {encoder_output.dtype}")

    # Create output variable
    print("\n3. Setting graph outputs...")

    # Create a new Variable for the output with explicit dtype
    context_output = gs.Variable(
        name="context_embedding",
        dtype=np.float32,
        shape=[226, "batch", 256]
    )

    # Find the node that produces the cut point and update its output
    for node in graph.nodes:
        for i, out in enumerate(node.outputs):
            if out.name == cut_point:
                # Keep the original output but also create our named output
                node.outputs[i] = context_output
                break

    # Set graph outputs
    graph.outputs = [context_output]

    # Cleanup: remove nodes not needed for output
    print("\n4. Cleaning up graph...")
    graph.cleanup()

    print(f"   After cleanup: {len(graph.nodes)} nodes")

    # Export
    print(f"\n5. Exporting to: {output_path}")

    model_out = gs.export_onnx(graph)

    # Set compatible IR version and opset
    model_out.ir_version = 9
    model_out.opset_import[0].version = 20  # Match original model (Gelu requires opset 20)

    # Run shape inference
    try:
        model_out = onnx.shape_inference.infer_shapes(model_out)
    except Exception as e:
        print(f"   Warning: Shape inference failed: {e}")

    onnx.save(model_out, output_path)

    # Verify
    model_verify = onnx.load(output_path)
    print(f"\n   Verification:")
    print(f"   - Nodes: {len(model_verify.graph.node)}")
    print(f"   - Inputs: {len(model_verify.graph.input)}")
    print(f"   - Outputs: {len(model_verify.graph.output)}")
    print(f"   - Output name: {model_verify.graph.output[0].name}")

    return True


def create_dit_core(model_path: str, output_path: str) -> bool:
    """
    Extract DiT core with dynamic timestep and context_embedding inputs.

    Key operations:
    1. Replace encoder output with a new input 'context_embedding'
    2. Replace hardcoded timestep constant with an input 'timestep'
    3. Output only DiT prediction (before solver math)
    4. Cleanup removes all encoder nodes

    Result: A model that takes context_embedding + sampled_trajectories + timestep
    and outputs dit_prediction, WITHOUT recomputing the encoder.
    """
    print("\n" + "=" * 80)
    print("Creating dit_core.onnx (WITHOUT encoder)")
    print("=" * 80)

    print(f"\n1. Loading model: {model_path}")
    model = onnx.load(model_path)

    # Detect version
    version = detect_model_version(model)
    print(f"   Detected model version: {version}")
    cut_points = get_cut_points(version)

    graph = gs.import_onnx(model)

    print(f"   Original nodes: {len(graph.nodes)}")

    # === Step 1: Replace encoder output with input ===
    encoder_output_name = cut_points["encoder_output"]

    print(f"\n2. Replacing encoder output with input...")
    print(f"   Cut point: {encoder_output_name}")

    # Create context_embedding input
    context_input = gs.Variable(
        name="context_embedding",
        dtype=np.float32,
        shape=[226, "batch", 256]
    )

    # Find all nodes that consume the encoder output
    encoder_output_tensor = None
    for tensor_name, tensor in graph.tensors().items():
        if tensor_name == encoder_output_name:
            encoder_output_tensor = tensor
            break

    if encoder_output_tensor is None:
        print(f"   ERROR: Encoder output tensor not found!")
        return False

    # Find and rewire all consumers
    consumers_rewired = 0
    for node in graph.nodes:
        for i, inp in enumerate(node.inputs):
            if hasattr(inp, 'name') and inp.name == encoder_output_name:
                node.inputs[i] = context_input
                consumers_rewired += 1

    print(f"   Rewired {consumers_rewired} consumer nodes to use context_embedding input")

    # Add context_embedding to graph inputs
    graph.inputs.append(context_input)

    # === Step 2: Replace timestep constant with input ===
    print(f"\n3. Replacing timestep constant with input...")

    timestep_constant_name = cut_points["timestep_const"]

    timestep_node = None
    for node in graph.nodes:
        if node.name == timestep_constant_name:
            timestep_node = node
            break

    if timestep_node is None:
        # Fallback: search by partial name match
        for node in graph.nodes:
            if "Constant_7" in node.name and "ego_encoder" in node.name:
                timestep_node = node
                break

    timestep_input = gs.Variable(
        name="timestep",
        dtype=np.float32,
        shape=[1]
    )

    if timestep_node:
        print(f"   Found timestep constant: {timestep_node.name}")

        old_output = timestep_node.outputs[0]
        old_output_name = old_output.name

        # Find all nodes that consume this constant
        consumers = []
        for node in graph.nodes:
            for i, inp in enumerate(node.inputs):
                if hasattr(inp, 'name') and inp.name == old_output_name:
                    consumers.append((node, i))

        for node, i in consumers:
            node.inputs[i] = timestep_input

        print(f"   Rewired {len(consumers)} consumers")

        # Mark constant for removal
        timestep_node.outputs = []

        graph.inputs.append(timestep_input)
        print(f"   Added 'timestep' as dynamic input")
    else:
        print(f"   WARNING: Timestep constant not found!")
        graph.inputs.append(timestep_input)

    # === Step 3: Set DiT output ===
    dit_output_name = cut_points["dit_output"]

    print(f"\n4. Setting output: {dit_output_name}")

    dit_output_var = gs.Variable(
        name="dit_prediction",
        dtype=np.float32,
        shape=["batch", 33, 81, 4]
    )

    for node in graph.nodes:
        for i, out in enumerate(node.outputs):
            if out.name == dit_output_name:
                node.outputs[i] = dit_output_var
                break

    graph.outputs = [dit_output_var]

    # === Step 4: Cleanup ===
    print(f"\n5. Cleaning up graph (removing encoder nodes)...")

    # Count before
    encoder_before = sum(1 for n in graph.nodes if '/encoder/' in n.name)
    print(f"   Encoder nodes before cleanup: {encoder_before}")

    graph.cleanup()

    # Count after
    encoder_after = sum(1 for n in graph.nodes if '/encoder/' in n.name)
    print(f"   Encoder nodes after cleanup: {encoder_after}")
    print(f"   Total nodes after cleanup: {len(graph.nodes)}")

    # === Step 5: Verify required inputs ===
    print(f"\n6. Analyzing required inputs...")

    # The graph inputs should now be:
    # - context_embedding (new)
    # - timestep (new)
    # - sampled_trajectories (original)
    # Plus any other inputs still needed by non-encoder parts

    required_inputs = set()
    for node in graph.nodes:
        for inp in node.inputs:
            if hasattr(inp, 'name') and inp in graph.inputs:
                required_inputs.add(inp.name)

    # Also include our new inputs
    required_inputs.add("context_embedding")
    required_inputs.add("timestep")

    print(f"   Required inputs: {len(required_inputs)}")

    # Filter graph inputs to only required ones
    new_inputs = []
    for inp in graph.inputs:
        if inp.name in required_inputs:
            new_inputs.append(inp)
            print(f"     - {inp.name}")

    graph.inputs = new_inputs

    # === Step 6: Export ===
    print(f"\n7. Exporting to: {output_path}")

    model_out = gs.export_onnx(graph)

    # Set compatible IR version and opset
    model_out.ir_version = 9
    model_out.opset_import[0].version = 20  # Match original model (Gelu requires opset 20)

    try:
        model_out = onnx.shape_inference.infer_shapes(model_out)
        print(f"   Shape inference OK")
    except Exception as e:
        print(f"   Shape inference skipped: {e}")

    onnx.save(model_out, output_path)

    # === Verification ===
    model_verify = onnx.load(output_path)

    # Count node types
    enc = sum(1 for n in model_verify.graph.node if '/encoder/' in n.name)
    dit = sum(1 for n in model_verify.graph.node if '/dit/' in n.name)

    print(f"\n   Verification:")
    print(f"   - Total nodes: {len(model_verify.graph.node)}")
    print(f"   - Encoder nodes: {enc} (should be 0 or minimal)")
    print(f"   - DiT nodes: {dit}")
    print(f"   - Inputs: {len(model_verify.graph.input)}")
    for inp in model_verify.graph.input:
        print(f"     - {inp.name}")
    print(f"   - Outputs: {[out.name for out in model_verify.graph.output]}")

    success = enc < 100  # Allow some small encoder-related nodes
    if success:
        print(f"\n   SUCCESS: Encoder effectively removed!")
    else:
        print(f"\n   WARNING: Encoder still present ({enc} nodes)")

    return success


def extract_turn_indicator_weights(backbone_path: str, output_path: str) -> bool:
    """
    Extract turn indicator MLP weights from diffusion_backbone.onnx.

    The MLP is: Linear(272 → 4)
    Input: concat(ego_traj[16], encoding_pooled[256])
    """
    print("\n" + "=" * 80)
    print("Extracting Turn Indicator Weights")
    print("=" * 80)

    import json
    from onnx import numpy_helper

    print(f"\n1. Loading backbone: {backbone_path}")

    if not Path(backbone_path).exists():
        print(f"   ERROR: Backbone not found!")
        return False

    model = onnx.load(backbone_path)

    weights = {}

    print(f"\n2. Searching for turn_indicator_predictor weights...")

    for init in model.graph.initializer:
        if 'turn_indicator_predictor' in init.name:
            arr = numpy_helper.to_array(init)
            print(f"   Found: {init.name} shape={arr.shape}")

            if 'weight' in init.name:
                weights['weight'] = arr.tolist()
            elif 'bias' in init.name:
                weights['bias'] = arr.tolist()

    if 'weight' not in weights or 'bias' not in weights:
        print(f"   ERROR: Could not find all required weights!")
        return False

    print(f"\n3. Saving to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)

    # Verify shapes
    w = np.array(weights['weight'])
    b = np.array(weights['bias'])
    print(f"   Weight shape: {w.shape}")  # Expected [4, 272]
    print(f"   Bias shape: {b.shape}")    # Expected [4]

    return True


def main():
    import os
    default_model_dir = os.environ.get(
        'DIFFUSION_PLANNER_MODEL_DIR',
        os.path.expanduser('~/autoware_data/diffusion_planner/v3.0'),
    )
    parser = argparse.ArgumentParser(description='GraphSurgeon ONNX Splitting')
    parser.add_argument('--onnx', type=str,
                        default=os.path.join(default_model_dir, 'diffusion_planner.onnx'),
                        help='Original ONNX model')
    parser.add_argument('--backbone', type=str,
                        default=os.path.join(default_model_dir, 'diffusion_planner.onnx'),
                        help='Model for turn indicator weights (can be same as --onnx)')
    parser.add_argument('--output-dir', type=str,
                        default=default_model_dir,
                        help='Output directory')
    parser.add_argument('--encoder-only', action='store_true',
                        help='Only create encoder')
    parser.add_argument('--dit-only', action='store_true',
                        help='Only create dit_core')
    parser.add_argument('--weights-only', action='store_true',
                        help='Only extract weights')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Detect version for naming
    model = onnx.load(args.onnx)
    version = detect_model_version(model)
    del model  # Free memory
    print(f"Detected model version: {version}")

    if not args.dit_only and not args.weights_only:
        encoder_path = str(output_dir / f"context_encoder_{version}.onnx")
        success &= create_context_encoder(args.onnx, encoder_path)

    if not args.encoder_only and not args.weights_only:
        dit_path = str(output_dir / "dit_core.onnx")
        success &= create_dit_core(args.onnx, dit_path)

    if not args.encoder_only and not args.dit_only:
        weights_path = str(output_dir / "turn_indicator_weights.json")
        success &= extract_turn_indicator_weights(args.backbone, weights_path)

    print("\n" + "=" * 80)
    if success:
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
    else:
        print("SOME OPERATIONS FAILED")
    print("=" * 80)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
