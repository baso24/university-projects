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

from environment import GRID_SIZE_8

class EmbeddingPuzzleNet(nn.Module):
    def __init__(self, grid_size=GRID_SIZE_8, embedding_dim=16, hidden_dims=[64, 32]):
        super(EmbeddingPuzzleNet, self).__init__()
        
        self.num_tiles = grid_size * grid_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # Tile Embedding Layer (9 tile IDs -> embedding_dim dimensional vector each)
        self.embedding = nn.Embedding(num_embeddings=self.num_tiles, embedding_dim=self.embedding_dim)
        
        # Flattened size: 9 * embedding_dim
        self.input_size = self.num_tiles * self.embedding_dim
        
        # Backward compatibility check
        if hidden_dims == [64, 32]:
            self.layer_1 = nn.Linear(in_features=self.input_size, out_features=64)
            self.layer_2 = nn.Linear(in_features=64, out_features=32)
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
            
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is long tensor of shape (batch_size, 9)
        embedded = self.embedding(x)  # shape: (batch_size, 9, embedding_dim)
        flattened = embedded.view(x.size(0), -1)  # shape: (batch_size, input_size)
        
        if not getattr(self, 'use_dynamic_layers', False):
            out = self.relu(self.layer_1(flattened))
            out = self.relu(self.layer_2(out))
        else:
            out = flattened
            for layer in self.layers:
                out = self.relu(layer(out))
                
        estimated_cost = self.output_layer(out)
        return estimated_cost

if __name__ == "__main__":
    model = EmbeddingPuzzleNet(grid_size=GRID_SIZE_8)
    sample_input = torch.tensor([[0, 7, 5, 8, 2, 4, 3, 6, 1]], dtype=torch.long)
    cost_pred = model(sample_input)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EmbeddingPuzzleNet initialized successfully!")
    print(f"Total trainable parameters: {total_params}")
    print(f"Output shape: {cost_pred.shape}")
