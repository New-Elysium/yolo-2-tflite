#!/usr/bin/env python
# coding=utf-8

"""
Generic Darknet CFG Parser
Supports parsing Darknet configuration files for YOLOv3, YOLOv4, YOLOv4-CSP, Scaled-YOLOv4, YOLOv7, etc.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional


class CFGParser:
    """Parse Darknet configuration files into structured format"""
    
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.net_config = {}
        self.layers = []
        self._parse_cfg()
    
    def _parse_cfg(self):
        """Parse the CFG file into network and layer configurations"""
        with open(self.cfg_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        current_layer = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers
            if line.startswith('[') and line.endswith(']'):
                # Save previous layer if exists
                if current_layer:
                    self.layers.append(current_layer)
                    current_layer = {}
                
                section_type = line[1:-1].strip()
                if section_type.lower() == 'net':
                    current_section = 'net'
                else:
                    current_section = 'layer'
                    current_layer['type'] = section_type.lower()
                continue
            
            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert to appropriate type
                value = self._convert_value(value)
                
                if current_section == 'net':
                    self.net_config[key] = value
                else:
                    current_layer[key] = value
        
        # Save the last layer
        if current_layer:
            self.layers.append(current_layer)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate Python type"""
        # Remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle boolean
        if value.lower() in ['true', '1']:
            return True
        elif value.lower() in ['false', '0']:
            return False
        
        # Handle lists (comma-separated)
        if ',' in value:
            try:
                return [self._convert_value(v.strip()) for v in value.split(',')]
            except ValueError:
                pass
        
        # Return as string
        return value
    
    def get_net_config(self) -> Dict[str, Any]:
        """Get network configuration"""
        return self.net_config
    
    def get_layers(self) -> List[Dict[str, Any]]:
        """Get layer configurations"""
        return self.layers
    
    def get_input_size(self) -> tuple:
        """Get input size from network config"""
        width = self.net_config.get('width', 416)
        height = self.net_config.get('height', 416)
        channels = self.net_config.get('channels', 3)
        return (height, width, channels)
    
    def get_num_classes(self) -> int:
        """Get number of classes from network config"""
        return self.net_config.get('classes', 80)
    
    def get_anchors(self) -> List[List[int]]:
        """Get anchors from network config"""
        anchors = self.net_config.get('anchors', [])
        if isinstance(anchors, list):
            # Reshape into list of [x, y] pairs
            anchor_pairs = []
            for i in range(0, len(anchors), 2):
                if i + 1 < len(anchors):
                    anchor_pairs.append([anchors[i], anchors[i+1]])
            return anchor_pairs
        return []
    
    def get_yolo_layers(self) -> List[Dict[str, Any]]:
        """Get YOLO detection layers"""
        return [layer for layer in self.layers if layer.get('type') == 'yolo']
    
    def get_convolutional_layers(self) -> List[Dict[str, Any]]:
        """Get convolutional layers"""
        return [layer for layer in self.layers if layer.get('type') == 'convolutional']
    
    def get_route_layers(self) -> List[Dict[str, Any]]:
        """Get route layers"""
        return [layer for layer in self.layers if layer.get('type') == 'route']
    
    def get_shortcut_layers(self) -> List[Dict[str, Any]]:
        """Get shortcut layers"""
        return [layer for layer in self.layers if layer.get('type') == 'shortcut']
    
    def get_upsample_layers(self) -> List[Dict[str, Any]]:
        """Get upsample layers"""
        return [layer for layer in self.layers if layer.get('type') == 'upsample']
    
    def get_maxpool_layers(self) -> List[Dict[str, Any]]:
        """Get maxpool layers"""
        return [layer for layer in self.layers if layer.get('type') == 'maxpool']
    
    def print_summary(self):
        """Print a summary of the parsed configuration"""
        print(f"Network Configuration: {self.net_config.get('net', 'unknown')}")
        print(f"Input size: {self.get_input_size()}")
        print(f"Number of classes: {self.get_num_classes()}")
        print(f"Total layers: {len(self.layers)}")
        
        layer_types = {}
        for layer in self.layers:
            layer_type = layer.get('type', 'unknown')
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        print("Layer types:")
        for layer_type, count in layer_types.items():
            print(f"  {layer_type}: {count}")


def parse_activation_function(activation: str) -> str:
    """Parse activation function name to standard format"""
    activation = activation.lower()
    activation_map = {
        'leaky': 'leaky',
        'mish': 'mish',
        'swish': 'swish',
        'linear': 'linear',
        'logistic': 'sigmoid',
        'relu': 'relu',
        'elu': 'elu',
        'selu': 'selu',
        'gelu': 'gelu',
        'hardmish': 'hardmish',
        'hardswish': 'hardswish'
    }
    return activation_map.get(activation, 'linear')


def parse_padding(padding: Any) -> str:
    """Parse padding parameter"""
    if isinstance(padding, int):
        if padding == 0:
            return 'valid'
        elif padding == 1:
            return 'same'
    elif isinstance(padding, str):
        return padding.lower()
    return 'same'


def get_layer_output_shape(input_shape: tuple, layer: Dict[str, Any]) -> tuple:
    """Calculate output shape for a given layer"""
    layer_type = layer.get('type')
    
    if layer_type == 'convolutional':
        kernel_size = layer.get('size', 1)
        stride = layer.get('stride', 1)
        padding = parse_padding(layer.get('pad', 0))
        
        if padding == 'same':
            h, w = input_shape[0], input_shape[1]
        else:  # valid
            h = (input_shape[0] - kernel_size) // stride + 1
            w = (input_shape[1] - kernel_size) // stride + 1
        
        filters = layer.get('filters', input_shape[2])
        return (h, w, filters)
    
    elif layer_type == 'maxpool':
        kernel_size = layer.get('size', 2)
        stride = layer.get('stride', 2)
        padding = parse_padding(layer.get('padding', 0))
        
        if padding == 'same':
            h = input_shape[0] // stride
            w = input_shape[1] // stride
        else:  # valid
            h = (input_shape[0] - kernel_size) // stride + 1
            w = (input_shape[1] - kernel_size) // stride + 1
        
        return (h, w, input_shape[2])
    
    elif layer_type == 'upsample':
        stride = layer.get('stride', 2)
        return (input_shape[0] * stride, input_shape[1] * stride, input_shape[2])
    
    elif layer_type == 'route':
        # Route layer concatenates previous layers
        # This is complex and depends on which layers are being routed
        return input_shape  # Placeholder
    
    elif layer_type == 'shortcut':
        # Shortcut layer adds previous layer
        return input_shape
    
    elif layer_type == 'yolo':
        # YOLO layer doesn't change spatial dimensions
        return input_shape
    
    # Default: return input shape unchanged
    return input_shape


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        parser = CFGParser(cfg_path)
        parser.print_summary()
    else:
        print("Usage: python cfg_parser.py <path_to_cfg_file>")
