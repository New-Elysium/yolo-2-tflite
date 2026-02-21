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
        self.weights_file = None
        self.weights_data = None
        self.position = 0
        
    def load_weights(self, model: keras.Model, weights_path: str) -> keras.Model:
        """Load weights from Darknet .weights file into the model"""
        self.weights_file = open(weights_path, 'rb')
        self.weights_data = self.weights_file.read()
        self.position = 0
        
        # Read header
        major, minor, revision, seen, _ = self._read_header()
        print(f"Darknet weights header: {major}.{minor}.{revision}, seen: {seen}")
        
        # Load weights for each layer
        conv_layers = [layer for layer in self.layers if layer.get('type') == 'convolutional']
        model_conv_layers = [layer for layer in model.layers if 'conv2d' in layer.name]
        
        conv_idx = 0
        bn_idx = 0
        
        for i, layer_config in enumerate(self.layers):
            if layer_config.get('type') == 'convolutional':
                if conv_idx < len(model_conv_layers):
                    self._load_convolutional_weights(
                        model_conv_layers[conv_idx], 
                        layer_config, 
                        bn_idx
                    )
                    conv_idx += 1
                    
                    if layer_config.get('batch_normalize', 1):
                        bn_idx += 1
        
        self.weights_file.close()
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
    
    def _load_convolutional_weights(self, conv_layer: keras.layers.Layer, 
                                layer_config: Dict[str, Any], bn_idx: int) -> None:
        """Load weights for a convolutional layer"""
        filters = layer_config.get('filters', 1)
        kernel_size = layer_config.get('size', 1)
        in_channels = layer_config.get('c', 3)
        batch_normalize = layer_config.get('batch_normalize', 1)
        groups = layer_config.get('groups', 1)
        
        # Read batch normalization weights if present
        if batch_normalize:
            bn_weights = self._read_weights(filters * 4)  # beta, gamma, mean, variance
            self._set_batch_norm_weights(conv_layer, bn_weights, bn_idx)
        
        # Read convolutional weights
        conv_shape = (filters, in_channels // groups, kernel_size, kernel_size)
        conv_weights = self._read_weights(np.prod(conv_shape))
        conv_weights = conv_weights.reshape(conv_shape)
        
        # Transpose from Darknet format (out, in, h, w) to TensorFlow format (h, w, in, out)
        conv_weights = np.transpose(conv_weights, (2, 3, 1, 0))
        
        # Read bias if no batch normalization
        if not batch_normalize:
            bias = self._read_weights(filters)
            self._set_conv_weights(conv_layer, conv_weights, bias)
        else:
            self._set_conv_weights(conv_layer, conv_weights, None)
    
    def _set_batch_norm_weights(self, conv_layer: keras.layers.Layer, 
                             bn_weights: np.ndarray, bn_idx: int) -> None:
        """Set batch normalization weights"""
        # Find the corresponding batch normalization layer
        bn_layer = None
        for layer in conv_layer._inbound_nodes[0].inbound_layers:
            if hasattr(layer, 'name') and 'batch_normalization' in layer.name:
                bn_layer = layer
                break
        
        if bn_layer is None:
            # Try to find BN layer in model
            model = conv_layer._get_node_by_index(0).layer._get_graph()
            for layer in model.layers:
                if hasattr(layer, 'name') and f'batch_normalization_{bn_idx}' in layer.name:
                    bn_layer = layer
                    break
        
        if bn_layer is not None:
            # Darknet order: [beta, gamma, mean, variance]
            # TensorFlow order: [gamma, beta, mean, variance]
            beta = bn_weights[0:filters]
            gamma = bn_weights[filters:2*filters]
            mean = bn_weights[2*filters:3*filters]
            variance = bn_weights[3*filters:4*filters]
            
            # Set weights in TensorFlow order
            bn_layer.set_weights([gamma, beta, mean, variance])
        else:
            print(f"Warning: Could not find batch normalization layer for conv layer")
    
    def _set_conv_weights(self, conv_layer: keras.layers.Layer, 
                        conv_weights: np.ndarray, bias: np.ndarray = None) -> None:
        """Set convolutional layer weights"""
        if bias is not None:
            conv_layer.set_weights([conv_weights, bias])
        else:
            conv_layer.set_weights([conv_weights])


class LegacyWeightsLoader:
    """Legacy weights loader for compatibility with existing code"""
    
    def __init__(self):
        pass
    
    def load_weights_legacy(self, model: keras.Model, weights_file: str, 
                        model_name: str = 'yolov4', is_tiny: bool = False) -> keras.Model:
        """Load weights using legacy layer index approach"""
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        
        # Determine layer configuration based on model type
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
                    # Read batch normalization weights
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]  # Reorder for TF
                    
                    try:
                        bn_layer = model.get_layer(bn_layer_name)
                        bn_layer.set_weights(bn_weights)
                    except ValueError:
                        print(f"Warning: Could not find batch normalization layer: {bn_layer_name}")
                    
                    j += 1
                else:
                    # Read bias
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                
                # Read convolutional weights
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
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


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 2:
        cfg_path = sys.argv[1]
        weights_path = sys.argv[2]
        
        # Build model
        from core.model_builder import build_model_from_cfg
        model = build_model_from_cfg(cfg_path)
        
        # Load weights
        model = load_weights_from_cfg(model, cfg_path, weights_path)
        print("Weights loaded successfully!")
    else:
        print("Usage: python weights_loader.py <cfg_path> <weights_path>")
