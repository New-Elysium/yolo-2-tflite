"""Convert YOLOv5 PyTorch weights (.pt) to numpy format (.npz).

This script extracts weights from a YOLOv5 PyTorch checkpoint and saves them
in a numpy archive (.npz) format that can be loaded without PyTorch installed.

Usage:
    python convert_weights.py --weights ./data/yolov5s.pt --output ./data/yolov5s.npz

Requirements:
    - PyTorch (pip install torch)
    - NumPy

The output .npz file can then be used with save_model.py:
    python save_model.py --weights ./data/yolov5s.npz --output ./checkpoints/yolov5s-640 \
        --input_size 640 --model yolov5
"""

import argparse
import numpy as np


def convert_pt_to_npz(weights_path, output_path):
    """Convert a YOLOv5 .pt checkpoint to .npz numpy archive.

    Args:
        weights_path: Path to the YOLOv5 .pt file.
        output_path: Path to save the .npz output.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. Install with:\n"
            "  pip install torch\n"
        )

    print(f"Loading PyTorch checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'model' in checkpoint:
        pt_model = checkpoint['model'].float()
    elif 'ema' in checkpoint and checkpoint['ema'] is not None:
        pt_model = checkpoint['ema'].float()
    else:
        raise ValueError("Could not find 'model' or 'ema' in checkpoint")

    state_dict = pt_model.state_dict()

    # Extract and organize weights
    weights = {}
    conv_idx = 0
    bn_idx = 0

    # Group keys by module
    processed_convs = set()
    processed_bns = set()

    for key in sorted(state_dict.keys()):
        if key.endswith('.conv.weight') and key not in processed_convs:
            processed_convs.add(key)
            pt_weight = state_dict[key].numpy()
            # Transpose from PyTorch (O, I, H, W) to TF (H, W, I, O)
            tf_weight = pt_weight.transpose(2, 3, 1, 0)
            weights[f'conv_{conv_idx}_weight'] = tf_weight

            # Check for bias (detection head layers)
            bias_key = key.replace('.conv.weight', '.conv.bias')
            if bias_key in state_dict:
                weights[f'conv_{conv_idx}_bias'] = state_dict[bias_key].numpy()

            # Check for BN
            bn_base = key.replace('.conv.weight', '.bn')
            bn_weight_key = f'{bn_base}.weight'
            if bn_weight_key in state_dict and bn_base not in processed_bns:
                processed_bns.add(bn_base)
                gamma = state_dict[f'{bn_base}.weight'].numpy()
                beta = state_dict[f'{bn_base}.bias'].numpy()
                mean = state_dict[f'{bn_base}.running_mean'].numpy()
                var = state_dict[f'{bn_base}.running_var'].numpy()
                weights[f'bn_{bn_idx}_weights'] = np.array([gamma, beta, mean, var], dtype=object)
                bn_idx += 1

            conv_idx += 1

    # Also handle detection head weights (model.24.m.X.weight/bias patterns)
    for key in sorted(state_dict.keys()):
        if '.m.' in key and key.endswith('.weight') and key not in processed_convs:
            if len(state_dict[key].shape) == 4:
                processed_convs.add(key)
                pt_weight = state_dict[key].numpy()
                tf_weight = pt_weight.transpose(2, 3, 1, 0)
                weights[f'conv_{conv_idx}_weight'] = tf_weight

                bias_key = key.replace('.weight', '.bias')
                if bias_key in state_dict:
                    weights[f'conv_{conv_idx}_bias'] = state_dict[bias_key].numpy()

                conv_idx += 1

    np.savez(output_path, **weights)
    print(f"Saved {conv_idx} conv layers and {bn_idx} BN layers to: {output_path}")
    print(f"File size: {np.os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Print model info if available
    if hasattr(pt_model, 'names'):
        names = pt_model.names
        if isinstance(names, dict):
            print(f"Classes: {len(names)} ({', '.join(list(names.values())[:5])}...)")
        elif isinstance(names, list):
            print(f"Classes: {len(names)} ({', '.join(names[:5])}...)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLOv5 PyTorch weights to numpy format')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to YOLOv5 .pt weights file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz path (default: same name as input)')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.weights.replace('.pt', '.npz')

    convert_pt_to_npz(args.weights, args.output)


if __name__ == '__main__':
    main()
