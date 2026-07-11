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

class PuzzleNet(nn.Module):
    def __init__(self, grid_size=GRID_SIZE_8, hidden_dims=[128, 64, 32]):
        super(PuzzleNet, self).__init__()
        
        # input size calculation for coordinate encoding
        # 8-puzzle (3x3): 9 tiles * 2 coordinates (row, col) = 18 input neurons
        # 15-puzzle (4x4): 16 tiles * 2 coordinates (row, col) = 32 input neurons
        self.num_tiles = grid_size * grid_size
        self.input_size = self.num_tiles * 2 
        self.hidden_dims = hidden_dims
        
        # To maintain backward compatibility with saved checkpoints, if hidden_dims matches the default,
        # we define the layers with the exact attribute names (layer_1, layer_2, layer_3, output_layer).
        if hidden_dims == [128, 64, 32]:
            self.layer_1 = nn.Linear(in_features=self.input_size, out_features=128)
            self.layer_2 = nn.Linear(in_features=128, out_features=64)
            self.layer_3 = nn.Linear(in_features=64, out_features=32)
            self.output_layer = nn.Linear(in_features=32, out_features=1)
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
        
        # relu
        self.relu = nn.ReLU()

    def forward(self, x):
        if not getattr(self, 'use_dynamic_layers', False):
            # pass through the first hidden layer + activation (relu)
            x = self.relu(self.layer_1(x))
            # pass through the second hidden layer + activation (relu)
            x = self.relu(self.layer_2(x))
            # pass through the third hidden layer + activation (relu)
            x = self.relu(self.layer_3(x))
        else:
            for layer in self.layers:
                x = self.relu(layer(x))
        
        # output layer
        estimated_cost = self.output_layer(x)
        
        return estimated_cost

# check model
if __name__ == "__main__":
    # instantiate the model for 8-puzzle
    # here we can choose the size of the puzzle (8 or 15)
    grid_size = GRID_SIZE_8
    model = PuzzleNet(grid_size=grid_size)
    print(model)