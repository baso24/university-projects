import os
import sys
import torch
import torch.nn as nn

# Add new_models and repository root to sys.path
_current = os.path.abspath(__file__)
while True:
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    if os.path.basename(_parent) == 'new_models':
        sys.path.append(_parent)
        sys.path.append(os.path.dirname(_parent))
        root_dir = _parent
        break
    _current = _parent

from environment import GRID_SIZE_8, GRID_SIZE_15

class SmallPuzzleNet(nn.Module):
    def __init__(self, grid_size=GRID_SIZE_8, hidden_dims=[48, 24]):
        super(SmallPuzzleNet, self).__init__()
        
        self.num_tiles = grid_size * grid_size
        self.input_size = self.num_tiles * 2 
        self.hidden_dims = hidden_dims
        
        # Backward compatibility check
        if hidden_dims == [48, 24]:
            self.layer_1 = nn.Linear(in_features=self.input_size, out_features=48)
            self.layer_2 = nn.Linear(in_features=48, out_features=24)
            self.output_layer = nn.Linear(in_features=24, out_features=1)
            self.use_dynamic_layers = False
        else:
            layers = []
            prev_dim = self.input_size
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                prev_dim = dim
            self.layers = nn.ModuleList(layers)
            self.output_layer = nn.Linear(prev_dim, 1)
            self.use_dynamic_layers = True
            
        self.relu = nn.ReLU()

    def forward(self, x):
        if not getattr(self, 'use_dynamic_layers', False):
            x = self.relu(self.layer_1(x))
            x = self.relu(self.layer_2(x))
        else:
            for layer in self.layers:
                x = self.relu(layer(x))
        estimated_cost = self.output_layer(x)
        return estimated_cost

if __name__ == "__main__":
    model = SmallPuzzleNet(grid_size=GRID_SIZE_8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SmallPuzzleNet initialized successfully!")
    print(f"Total trainable parameters: {total_params}")
