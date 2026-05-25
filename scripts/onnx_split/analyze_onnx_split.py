#!/usr/bin/env python3
"""
ONNX Model Analysis Tool for Finding Encoder-Decoder Split Point

This script analyzes the existing diffusion_planner.onnx to find the
cut point between Context Encoder and Denoising Core.
"""

import onnx
import onnx.numpy_helper
from onnx import shape_inference
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path


def analyze_onnx_model(onnx_path, verbose=False):
    """
    Analyze ONNX model structure to find split points.

    Args:
        onnx_path: Path to ONNX model file
        verbose: Print detailed node information
    """
    print("="*80)
    print(f"ANALYZING: {onnx_path}")
    print("="*80)

    # Load model
    print("\n📦 Loading ONNX model...")
    model = onnx.load(onnx_path)

    # Infer shapes
    print("🔍 Inferring tensor shapes...")
    model = shape_inference.infer_shapes(model)

    graph = model.graph

    print(f"\n✅ Model loaded successfully")
    print(f"   IR version: {model.ir_version}")
    print(f"   Producer: {model.producer_name} {model.producer_version}")
    print(f"   Opset: {model.opset_import[0].version}")
    print(f"   Graph nodes: {len(graph.node)}")

    # ==================================================================
    # 1. Analyze Inputs
    # ==================================================================
    print("\n" + "="*80)
    print("📥 MODEL INPUTS")
    print("="*80)

    for idx, inp in enumerate(graph.input):
        shape = get_tensor_shape(inp)
        dtype = get_tensor_dtype(inp)
        print(f"{idx:3d}. {inp.name:40s} {str(shape):30s} {dtype}")

    # ==================================================================
    # 2. Analyze Outputs
    # ==================================================================
    print("\n" + "="*80)
    print("📤 MODEL OUTPUTS")
    print("="*80)

    for idx, out in enumerate(graph.output):
        shape = get_tensor_shape(out)
        dtype = get_tensor_dtype(out)
        print(f"{idx:3d}. {out.name:40s} {str(shape):30s} {dtype}")

    # ==================================================================
    # 3. Build node graph
    # ==================================================================
    print("\n🔧 Building computation graph...")

    # Map: tensor_name -> list of nodes that produce it
    producers = {}  # tensor_name -> node_name
    consumers = defaultdict(list)  # tensor_name -> [node_names]

    for node in graph.node:
        # Outputs
        for output in node.output:
            producers[output] = node.name or f"{node.op_type}_{node.output[0]}"

        # Inputs
        for input_name in node.input:
            consumers[input_name].append(node.name or f"{node.op_type}_{node.output[0]}")

    # ==================================================================
    # 4. Find intermediate large tensors (potential cut points)
    # ==================================================================
    print("\n" + "="*80)
    print("🎯 POTENTIAL CUT POINTS (Large Intermediate Tensors)")
    print("="*80)

    # Get all value_info (intermediate tensors)
    intermediate_tensors = {}
    for value_info in graph.value_info:
        shape = get_tensor_shape(value_info)
        if shape:
            numel = np.prod([d for d in shape if isinstance(d, int)])
            intermediate_tensors[value_info.name] = {
                'shape': shape,
                'dtype': get_tensor_dtype(value_info),
                'numel': numel
            }

    # Sort by size (large tensors are likely important intermediate results)
    sorted_tensors = sorted(intermediate_tensors.items(), key=lambda x: x[1]['numel'], reverse=True)

    print("\nTop 30 largest intermediate tensors:")
    print(f"{'Rank':>4} {'Tensor Name':<50} {'Shape':<30} {'Elements':>12} {'Producer':<30}")
    print("-" * 130)

    for rank, (name, info) in enumerate(sorted_tensors[:30], 1):
        producer = producers.get(name, "INPUT")
        print(f"{rank:4d}. {name:<50} {str(info['shape']):<30} {info['numel']:12,d} {producer:<30}")

    # ==================================================================
    # 5. Analyze Context/Embedding-like tensors
    # ==================================================================
    print("\n" + "="*80)
    print("🧠 CONTEXT/EMBEDDING CANDIDATES (Heuristic Search)")
    print("="*80)

    # Heuristics for finding context embedding:
    # - Name contains: context, embed, encoding, hidden, feature
    # - Shape: [batch, seq_len, hidden_dim] where hidden_dim ~ 256-512
    # - Located in middle of graph

    context_candidates = []

    for name, info in intermediate_tensors.items():
        shape = info['shape']

        # Check if shape matches [B, N, D] pattern
        if len(shape) == 3:
            if isinstance(shape[2], int) and 128 <= shape[2] <= 512:
                score = 0

                # Name-based scoring
                keywords = ['context', 'embed', 'encoding', 'hidden', 'feature', 'condition']
                name_lower = name.lower()
                for kw in keywords:
                    if kw in name_lower:
                        score += 2

                # Position in graph (middle = higher score)
                if name in producers:
                    producer_node = next((n for n in graph.node if n.name == producers[name]), None)
                    if producer_node:
                        node_idx = list(graph.node).index(producer_node)
                        middle_ratio = abs(node_idx / len(graph.node) - 0.5)
                        score += (1 - middle_ratio) * 3

                context_candidates.append({
                    'name': name,
                    'shape': shape,
                    'score': score,
                    'producer': producers.get(name, 'INPUT')
                })

    # Sort by score
    context_candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"{'Rank':>4} {'Score':>6} {'Tensor Name':<50} {'Shape':<25} {'Producer':<30}")
    print("-" * 120)

    for rank, cand in enumerate(context_candidates[:20], 1):
        print(f"{rank:4d}. {cand['score']:6.2f} {cand['name']:<50} {str(cand['shape']):<25} {cand['producer']:<30}")

    # ==================================================================
    # 6. Find nodes connected to sampled_trajectories (denoising input)
    # ==================================================================
    print("\n" + "="*80)
    print("🎲 DENOISING BACKBONE ENTRY POINTS")
    print("="*80)

    # Find sampled_trajectories input
    sampled_traj_input = None
    for inp in graph.input:
        if 'sampled' in inp.name.lower() or 'noise' in inp.name.lower():
            sampled_traj_input = inp.name
            break

    if sampled_traj_input:
        print(f"\nFound denoising input: {sampled_traj_input}")
        print(f"Consumers of {sampled_traj_input}:")

        for consumer in consumers.get(sampled_traj_input, []):
            # Find the node
            node = next((n for n in graph.node if n.name == consumer or consumer in str(n.output)), None)
            if node:
                print(f"  - Node: {node.op_type:20s} {consumer}")
                print(f"    Inputs: {node.input}")
                print(f"    Outputs: {node.output}")
    else:
        print("⚠️  Could not find sampled_trajectories input")

    # ==================================================================
    # 7. Detailed node analysis (optional)
    # ==================================================================
    if verbose:
        print("\n" + "="*80)
        print("📊 DETAILED NODE ANALYSIS")
        print("="*80)

        for idx, node in enumerate(graph.node):
            print(f"\nNode {idx:4d}: {node.op_type:20s} {node.name or '(unnamed)'}")
            print(f"  Inputs:  {node.input}")
            print(f"  Outputs: {node.output}")

    # ==================================================================
    # 8. Recommendations
    # ==================================================================
    print("\n" + "="*80)
    print("💡 SPLIT POINT RECOMMENDATIONS")
    print("="*80)

    print("\nBased on analysis, likely split points:")

    if context_candidates:
        top_candidate = context_candidates[0]
        print(f"\n🎯 PRIMARY RECOMMENDATION:")
        print(f"   Cut Point: {top_candidate['name']}")
        print(f"   Shape: {top_candidate['shape']}")
        print(f"   Confidence: {top_candidate['score']:.2f}/10")
        print(f"\n   This tensor should be:")
        print(f"   - Output of context_encoder.onnx")
        print(f"   - Input to denoise_core.onnx (along with x_t and timestep)")

    print("\n📋 Next steps:")
    print("   1. Verify the recommended cut point by inspecting the graph")
    print("   2. Use split_onnx_model.py to perform the actual split")
    print("   3. Test both sub-models independently")

    print("\n" + "="*80)

    return {
        'model': model,
        'intermediate_tensors': intermediate_tensors,
        'context_candidates': context_candidates,
        'producers': producers,
        'consumers': consumers,
    }


def get_tensor_shape(tensor_proto):
    """Extract shape from tensor proto."""
    if hasattr(tensor_proto, 'type') and hasattr(tensor_proto.type, 'tensor_type'):
        shape = []
        for dim in tensor_proto.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append('?')
        return shape
    return None


def get_tensor_dtype(tensor_proto):
    """Extract dtype from tensor proto."""
    if hasattr(tensor_proto, 'type') and hasattr(tensor_proto.type, 'tensor_type'):
        dtype_map = {
            1: 'float32',
            2: 'uint8',
            3: 'int8',
            5: 'int16',
            6: 'int32',
            7: 'int64',
            9: 'bool',
            10: 'float16',
            11: 'double',
        }
        dtype_int = tensor_proto.type.tensor_type.elem_type
        return dtype_map.get(dtype_int, f'unknown({dtype_int})')
    return 'unknown'


def visualize_graph_structure(onnx_path, output_path=None):
    """
    Create a simple visualization of the graph structure.
    Requires netron or graphviz (optional).
    """
    try:
        import netron
        print(f"\n🌐 Starting Netron visualization server...")
        print(f"   Open your browser to inspect the model graph")
        netron.start(onnx_path)
    except ImportError:
        print("\n⚠️  Netron not installed. Install with: pip install netron")
        print("   Then run: netron diffusion_planner.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze ONNX model to find encoder-decoder split point'
    )
    parser.add_argument('onnx_file', type=str,
                        help='Path to ONNX model file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed node information')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch Netron visualization (requires netron package)')

    args = parser.parse_args()

    # Analyze
    results = analyze_onnx_model(args.onnx_file, args.verbose)

    # Visualize (optional)
    if args.visualize:
        visualize_graph_structure(args.onnx_file)

    # Save analysis results
    output_file = Path(args.onnx_file).parent / 'analysis_results.json'
    print(f"\n💾 Saving analysis results to: {output_file}")

    import json
    results_serializable = {
        'context_candidates': [
            {
                'name': c['name'],
                'shape': [str(s) for s in c['shape']],
                'score': c['score'],
                'producer': c['producer']
            }
            for c in results['context_candidates'][:10]
        ],
        'num_nodes': len(results['model'].graph.node),
        'num_inputs': len(results['model'].graph.input),
        'num_outputs': len(results['model'].graph.output),
    }

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("✅ Analysis complete!")
