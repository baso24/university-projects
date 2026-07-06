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

class MultiTaskDeltaPuzzleNet(nn.Module):
    def __init__(self, grid_size=GRID_SIZE_8, hidden_dims=[128, 64, 32]):
        super(MultiTaskDeltaPuzzleNet, self).__init__()
        
        self.num_tiles = grid_size * grid_size
        self.input_size = self.num_tiles * 2 
        self.hidden_dims = hidden_dims
        
        # Backward compatibility check
        if hidden_dims == [128, 64, 32]:
            self.shared_layer_1 = nn.Linear(in_features=self.input_size, out_features=128)
            self.shared_layer_2 = nn.Linear(in_features=128, out_features=64)
            self.shared_layer_3 = nn.Linear(in_features=64, out_features=32)
            self.cost_head = nn.Linear(in_features=32, out_features=1)
            self.manhattan_head = nn.Linear(in_features=32, out_features=1)
            self.use_dynamic_layers = False
        else:
            layers = []
            prev_dim = self.input_size
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                prev_dim = dim
            self.layers = nn.ModuleList(layers)
            self.cost_head = nn.Linear(in_features=prev_dim, out_features=1)
            self.manhattan_head = nn.Linear(in_features=prev_dim, out_features=1)
            self.use_dynamic_layers = True
        
        self.relu = nn.ReLU()

    def forward(self, x, return_aux=False):
        if not getattr(self, 'use_dynamic_layers', False):
            x = self.relu(self.shared_layer_1(x))
            x = self.relu(self.shared_layer_2(x))
            x = self.relu(self.shared_layer_3(x))
        else:
            for layer in self.layers:
                x = self.relu(layer(x))
        
        cost_pred = self.cost_head(x)
        
        if return_aux:
            manhattan_pred = self.manhattan_head(x)
            return cost_pred, manhattan_pred
        
        return cost_pred

if __name__ == "__main__":
    model = MultiTaskDeltaPuzzleNet(grid_size=GRID_SIZE_8)
    sample_input = torch.randn(1, 18)
    cost, manhattan = model(sample_input, return_aux=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MultiTaskDeltaPuzzleNet initialized successfully!")
    print(f"Total trainable parameters: {total_params}")
    print(f"Cost Output shape: {cost.shape} | Manhattan Output shape: {manhattan.shape}")
