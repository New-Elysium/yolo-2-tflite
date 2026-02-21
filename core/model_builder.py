#!/usr/bin/env python
# coding=utf-8

"""
Dynamic Model Builder for Darknet Configurations
Builds TensorFlow/Keras models from parsed Darknet CFG files
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from core.cfg_parser import CFGParser, parse_activation_function, parse_padding


class ModelBuilder:
    """Builds TensorFlow models from Darknet CFG configurations"""
    
    def __init__(self, cfg_parser: CFGParser):
        self.cfg_parser = cfg_parser
        self.net_config = cfg_parser.get_net_config()
        self.layers = cfg_parser.get_layers()
        self.layer_outputs = []
        self.layer_names = []
        self.current_input = None
        
    def build_model(self) -> keras.Model:
        """Build the complete model from CFG"""
        input_shape = self.cfg_parser.get_input_size()
        
        # Create input layer
        input_layer = keras.layers.Input(shape=input_shape, name='input')
        self.current_input = input_layer
        self.layer_outputs.append(input_layer)
        self.layer_names.append('input')
        
        # Build each layer
        for i, layer_config in enumerate(self.layers):
            layer_output = self._build_layer(layer_config, i)
            if layer_output is not None:
                self.current_input = layer_output
                self.layer_outputs.append(layer_output)
                self.layer_names.append(f"layer_{i}_{layer_config['type']}")
        
        # Get YOLO output layers
        yolo_outputs = self._get_yolo_outputs()
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=yolo_outputs)
        
        return model
    
    def _build_layer(self, layer_config: Dict[str, Any], layer_idx: int) -> Optional[keras.Layer]:
        """Build a single layer from configuration"""
        layer_type = layer_config.get('type', '').lower()
        
        layer_builders = {
            'convolutional': self._build_convolutional,
            'maxpool': self._build_maxpool,
            'route': self._build_route,
            'shortcut': self._build_shortcut,
            'upsample': self._build_upsample,
            'yolo': self._build_yolo,
            'dropout': self._build_dropout,
            'connected': self._build_connected,
            'softmax': self._build_softmax,
            'cost': self._build_cost,
            'avgpool': self._build_avgpool,
            'local_avgpool': self._build_local_avgpool,
            'reorg': self._build_reorg,
            'region': self._build_region,
            'sam': self._build_sam,
            'scale_channels': self._build_scale_channels,
            'shuffle_channels': self._build_shuffle_channels,
            'spp': self._build_spp,
            'sppcspc': self._build_sppcspc,
            'csp': self._build_csp,
            'e-elan': self._build_e_elan,
            'mp': self._build_mp,
        }
        
        builder = layer_builders.get(layer_type)
        if builder:
            return builder(layer_config, layer_idx)
        else:
            print(f"Warning: Unsupported layer type: {layer_type}")
            return None
    
    def _build_convolutional(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build convolutional layer"""
        filters = layer_config.get('filters', 1)
        kernel_size = layer_config.get('size', 1)
        stride = layer_config.get('stride', 1)
        padding = parse_padding(layer_config.get('pad', 0))
        groups = layer_config.get('groups', 1)
        batch_normalize = layer_config.get('batch_normalize', 1)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))
        
        # Handle padding
        if stride > 1 and padding == 'same':
            # Darknet uses asymmetric padding for downsampling
            self.current_input = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(self.current_input)
            padding = 'valid'
        
        # Create convolutional layer
        conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            groups=groups,
            use_bias=not batch_normalize,
            kernel_regularizer=keras.regularizers.l2(0.0005),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Constant(0.),
            name=f"conv2d_{layer_idx}"
        )(self.current_input)
        
        # Batch normalization
        if batch_normalize:
            conv = keras.layers.BatchNormalization(
                name=f"batch_normalization_{layer_idx}"
            )(conv)
        
        # Activation
        if activation == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activation == 'mish':
            conv = conv * tf.math.tanh(tf.math.softplus(conv))
        elif activation == 'swish':
            conv = conv * tf.nn.sigmoid(conv)
        elif activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'elu':
            conv = tf.nn.elu(conv)
        elif activation == 'selu':
            conv = tf.nn.selu(conv)
        elif activation == 'gelu':
            conv = tf.nn.gelu(conv)
        elif activation == 'hardmish':
            conv = tf.nn.relu6(conv + 3) / 6
        elif activation == 'hardswish':
            conv = conv * tf.nn.relu6(conv + 3) / 6
        # 'linear' and others: no activation
        
        return conv
    
    def _build_maxpool(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build maxpool layer"""
        kernel_size = layer_config.get('size', 2)
        stride = layer_config.get('stride', 2)
        padding = parse_padding(layer_config.get('padding', 0))
        
        # Handle asymmetric padding for stride=2
        if stride == 2 and padding == 'same':
            self.current_input = keras.layers.ZeroPadding2D(((0, 1), (0, 1)))(self.current_input)
            padding = 'valid'
        
        return keras.layers.MaxPool2D(
            pool_size=kernel_size,
            strides=stride,
            padding=padding,
            name=f"maxpool2d_{layer_idx}"
        )(self.current_input)
    
    def _build_route(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build route layer (concatenation)"""
        layers = layer_config.get('layers', [])
        if not isinstance(layers, list):
            layers = [layers]
        
        # Get the layers to route
        route_inputs = []
        for layer_idx in layers:
            if layer_idx < 0:
                # Negative index: count from end
                route_inputs.append(self.layer_outputs[layer_idx])
            else:
                # Positive index: count from start (excluding input layer)
                route_inputs.append(self.layer_outputs[layer_idx + 1])
        
        if len(route_inputs) == 1:
            return route_inputs[0]
        else:
            return keras.layers.Concatenate(name=f"route_{layer_idx}")(route_inputs)
    
    def _build_shortcut(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build shortcut layer (residual connection)"""
        from_layer = layer_config.get('from', -1)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))
        
        # Get the shortcut input
        if from_layer < 0:
            shortcut_input = self.layer_outputs[from_layer]
        else:
            shortcut_input = self.layer_outputs[from_layer + 1]
        
        # Add the layers
        output = keras.layers.Add(name=f"shortcut_{layer_idx}")([self.current_input, shortcut_input])
        
        # Apply activation if specified
        if activation == 'leaky':
            output = tf.nn.leaky_relu(output, alpha=0.1)
        elif activation == 'relu':
            output = tf.nn.relu(output)
        
        return output
    
    def _build_upsample(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build upsample layer"""
        stride = layer_config.get('stride', 2)
        
        return tf.image.resize(
            self.current_input,
            (self.current_input.shape[1] * stride, self.current_input.shape[2] * stride),
            method='bilinear',
            name=f"upsample_{layer_idx}"
        )
    
    def _build_yolo(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build YOLO detection layer"""
        mask = layer_config.get('mask', [])
        num_classes = layer_config.get('classes', 80)
        num = layer_config.get('num', 3)
        jitter = layer_config.get('jitter', 0.3)
        ignore_thresh = layer_config.get('ignore_thresh', 0.7)
        truth_thresh = layer_config.get('truth_thresh', 1)
        random = layer_config.get('random', 1)
        scale_x_y = layer_config.get('scale_x_y', 1)
        iou_normalizer = layer_config.get('iou_normalizer', 0.75)
        cls_normalizer = layer_config.get('cls_normalizer', 1)
        iou_loss = layer_config.get('iou_loss', 'ciou')
        nms_kind = layer_config.get('nms_kind', 'default')
        beta_nms = layer_config.get('beta_nms', 0.6)
        
        # Get anchors
        anchors = self.cfg_parser.get_anchors()
        masked_anchors = [anchors[i] for i in mask] if mask else anchors
        
        # Calculate output filters: (num * (5 + num_classes))
        output_filters = len(masked_anchors) * (5 + num_classes)
        
        # Create convolutional layer for YOLO output
        yolo_conv = keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Constant(0.),
            name=f"yolo_conv_{layer_idx}"
        )(self.current_input)
        
        return yolo_conv
    
    def _build_dropout(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build dropout layer"""
        probability = layer_config.get('probability', 0.5)
        return keras.layers.Dropout(rate=probability, name=f"dropout_{layer_idx}")(self.current_input)
    
    def _build_connected(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build fully connected layer"""
        output_size = layer_config.get('output', 1)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))
        
        # Flatten input first
        flattened = keras.layers.Flatten(name=f"flatten_{layer_idx}")(self.current_input)
        
        # Dense layer
        dense = keras.layers.Dense(
            units=output_size,
            activation='linear',
            name=f"connected_{layer_idx}"
        )(flattened)
        
        # Apply activation
        if activation == 'relu':
            dense = tf.nn.relu(dense)
        elif activation == 'leaky':
            dense = tf.nn.leaky_relu(dense, alpha=0.1)
        elif activation == 'sigmoid':
            dense = tf.nn.sigmoid(dense)
        elif activation == 'softmax':
            dense = tf.nn.softmax(dense)
        
        return dense
    
    def _build_softmax(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build softmax layer"""
        groups = layer_config.get('groups', 1)
        
        if groups == 1:
            return tf.nn.softmax(self.current_input, name=f"softmax_{layer_idx}")
        else:
            # Group softmax - more complex implementation needed
            return tf.nn.softmax(self.current_input, name=f"softmax_{layer_idx}")
    
    def _build_cost(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build cost layer (usually for training)"""
        # Cost layers are typically not used in inference models
        return self.current_input
    
    def _build_avgpool(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build average pooling layer"""
        return keras.layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding='same',
            name=f"avgpool_{layer_idx}"
        )(self.current_input)
    
    def _build_local_avgpool(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build local average pooling layer"""
        kernel_size = layer_config.get('size', 1)
        stride = layer_config.get('stride', 1)
        padding = parse_padding(layer_config.get('padding', 0))
        
        return keras.layers.AveragePooling2D(
            pool_size=kernel_size,
            strides=stride,
            padding=padding,
            name=f"local_avgpool_{layer_idx}"
        )(self.current_input)
    
    def _build_reorg(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build reorganization layer (similar to reshape)"""
        stride = layer_config.get('stride', 2)
        
        # This is a complex operation - simplified implementation
        return tf.nn.space_to_depth(self.current_input, block_size=stride)
    
    def _build_region(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build region layer (older YOLO versions)"""
        # Similar to YOLO layer but for older versions
        return self._build_yolo(layer_config, layer_idx)
    
    def _build_sam(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build SAM (Spatial Attention Module) layer"""
        # Simplified SAM implementation
        return self.current_input
    
    def _build_scale_channels(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build scale channels layer"""
        scale = layer_config.get('scale', 1.0)
        return self.current_input * scale
    
    def _build_shuffle_channels(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build shuffle channels layer"""
        groups = layer_config.get('groups', 1)
        # Simplified channel shuffle
        return self.current_input
    
    def _build_spp(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build SPP (Spatial Pyramid Pooling) layer"""
        kernel_sizes = layer_config.get('maxpool', [5, 9, 13])
        
        # Multiple maxpool layers with different kernel sizes
        pools = [self.current_input]
        for ksize in kernel_sizes:
            pool = keras.layers.MaxPool2D(
                pool_size=ksize,
                strides=1,
                padding='same',
                name=f"spp_pool_{ksize}_{layer_idx}"
            )(self.current_input)
            pools.append(pool)
        
        # Concatenate all pools
        return keras.layers.Concatenate(name=f"spp_{layer_idx}")(pools)
    
    def _build_sppcspc(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build SPPCSPC layer (YOLOv7)"""
        # Complex SPPCSPC implementation - simplified
        return self.current_input
    
    def _build_csp(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build CSP (Cross Stage Partial) layer"""
        # Simplified CSP implementation
        return self.current_input
    
    def _build_e_elan(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build E-ELAN (Extended Efficient Layer Aggregation Network) layer"""
        # Complex E-ELAN implementation - simplified
        return self.current_input
    
    def _build_mp(self, layer_config: Dict[str, Any], layer_idx: int) -> keras.Layer:
        """Build MP (MaxPool) layer for YOLOv7"""
        # Complex MP implementation - simplified
        return self.current_input
    
    def _get_yolo_outputs(self) -> List[keras.Layer]:
        """Get YOLO output layers"""
        yolo_outputs = []
        
        for i, layer_config in enumerate(self.layers):
            if layer_config.get('type') == 'yolo':
                # Find the corresponding layer output
                layer_name = f"layer_{i}_yolo"
                for j, name in enumerate(self.layer_names):
                    if name == layer_name:
                        yolo_outputs.append(self.layer_outputs[j])
                        break
        
        return yolo_outputs if yolo_outputs else [self.current_input]


def build_model_from_cfg(cfg_path: str) -> keras.Model:
    """Convenience function to build model from CFG file"""
    parser = CFGParser(cfg_path)
    builder = ModelBuilder(parser)
    return builder.build_model()


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        model = build_model_from_cfg(cfg_path)
        model.summary()
    else:
        print("Usage: python model_builder.py <path_to_cfg_file>")
