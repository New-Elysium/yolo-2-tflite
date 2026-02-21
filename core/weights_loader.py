#!/usr/bin/env python
# coding=utf-8

"""
Generic Darknet Weights Loader
Loads Darknet .weights files into dynamically built TensorFlow models
"""

import numpy as np
import tensorflow as tf
import keras
from typing import Dict, List, Any, Tuple
from core.cfg_parser import CFGParser


class WeightsLoader:
    """Loads Darknet weights into TensorFlow models"""

    def __init__(self, cfg_parser: CFGParser):
        self.cfg_parser = cfg_parser
        self.layers = cfg_parser.get_layers()
        self.weights_data = None
        self.position = 0

    def load_weights(self, model: keras.Model, weights_path: str) -> keras.Model:
        """Load weights from Darknet .weights file into the model"""
        with open(weights_path, 'rb') as f:
            self.weights_data = f.read()
        self.position = 0

        # Read header
        major, minor, revision, seen, _ = self._read_header()
        print(f"Darknet weights header: {major}.{minor}.{revision}, seen: {seen}")

        # Build a map of model layers by name for reliable lookup
        model_layers_by_name = {layer.name: layer for layer in model.layers}

        # Iterate through CFG layers and load weights for convolutional layers
        for i, layer_config in enumerate(self.layers):
            if layer_config.get('type') != 'convolutional':
                continue

            conv_name = f"conv2d_{i}"
            bn_name = f"batch_normalization_{i}"

            conv_layer = model_layers_by_name.get(conv_name)
            if conv_layer is None:
                print(f"Warning: Conv layer {conv_name} not found in model, skipping")
                continue

            batch_normalize = layer_config.get('batch_normalize', 0)
            filters = layer_config.get('filters', 1)
            kernel_size = layer_config.get('size', 1)

            # Get actual input channels from the built model layer
            in_channels = conv_layer.input_shape[-1]
            groups = layer_config.get('groups', 1)

            if batch_normalize:
                # Darknet order: [beta, gamma, mean, variance] then [conv_weights]
                bn_weights = self._read_weights(filters * 4)
                bn_weights = bn_weights.reshape((4, filters))
                # Darknet: [beta, gamma, mean, variance]
                # TF:      [gamma, beta, mean, variance]
                bn_weights_tf = bn_weights[[1, 0, 2, 3]]

                bn_layer = model_layers_by_name.get(bn_name)
                if bn_layer is not None:
                    bn_layer.set_weights(bn_weights_tf)
                else:
                    print(f"Warning: BN layer {bn_name} not found in model")
            else:
                # Darknet order for non-BN: [bias] then [conv_weights]
                conv_bias = self._read_weights(filters)

            # Read convolutional weights (always after BN/bias)
            conv_shape = (filters, in_channels // groups, kernel_size, kernel_size)
            conv_weights = self._read_weights(np.prod(conv_shape))
            # Darknet: (out, in, h, w) -> TF: (h, w, in, out)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if batch_normalize:
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        remaining = len(self.weights_data) - self.position
        print(f"Weights loaded. Remaining bytes: {remaining}")
        return model

    def _read_header(self) -> Tuple[int, int, int, int, int]:
        """Read Darknet weights file header"""
        header = np.frombuffer(self.weights_data[self.position:self.position+20], dtype=np.int32)
        self.position += 20
        return tuple(header)

    def _read_weights(self, count: int) -> np.ndarray:
        """Read weights from current position"""
        weights = np.frombuffer(
            self.weights_data[self.position:self.position+count*4],
            dtype=np.float32
        )
        self.position += count * 4
        return weights


class LegacyWeightsLoader:
    """Legacy weights loader for compatibility with existing code"""

    def load_weights_legacy(self, model: keras.Model, weights_file: str,
                        model_name: str = 'yolov4', is_tiny: bool = False) -> keras.Model:
        """Load weights using legacy layer index approach"""
        if is_tiny:
            if model_name == 'yolov3':
                layer_size = 13
                output_pos = [9, 12]
            else:
                layer_size = 21
                output_pos = [17, 20]
        else:
            if model_name == 'yolov3':
                layer_size = 75
                output_pos = [58, 66, 74]
            else:
                layer_size = 110
                output_pos = [93, 101, 109]

        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(layer_size):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

            try:
                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in output_pos:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    bn_layer.set_weights(bn_weights)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in output_pos:
                    conv_layer.set_weights([conv_weights])
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            except ValueError as e:
                print(f"Warning: Could not load weights for layer {conv_layer_name}: {e}")
                continue

        wf.close()
        return model


def load_weights_from_cfg(model: keras.Model, cfg_path: str, weights_path: str) -> keras.Model:
    """Convenience function to load weights using CFG-based approach"""
    cfg_parser = CFGParser(cfg_path)
    loader = WeightsLoader(cfg_parser)
    return loader.load_weights(model, weights_path)


def load_weights_legacy(model: keras.Model, weights_path: str,
                     model_name: str = 'yolov4', is_tiny: bool = False) -> keras.Model:
    """Convenience function to load weights using legacy approach"""
    loader = LegacyWeightsLoader()
    return loader.load_weights_legacy(model, weights_path, model_name, is_tiny)
