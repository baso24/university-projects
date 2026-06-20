import torch
import torch.nn as nn
from environment import GRID_SIZE_8, GRID_SIZE_15

class PuzzleNet(nn.Module):
    def __init__(self, grid_size):
        super(PuzzleNet, self).__init__()
        
        # input size calculation for coordinate encoding
        # 8-puzzle (3x3): 9 tiles * 2 coordinates (row, col) = 18 input neurons
        # 15-puzzle (4x4): 16 tiles * 2 coordinates (row, col) = 32 input neurons
        self.num_tiles = grid_size * grid_size
        self.input_size = self.num_tiles * 2 
        
        # architecture
        self.layer_1 = nn.Linear(in_features=self.input_size, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=64)
        self.layer_3 = nn.Linear(in_features=64, out_features=32)
        
        # output layer
        self.output_layer = nn.Linear(in_features=32, out_features=1)\
        
        # relu
        self.relu = nn.ReLU()

    def forward(self, x):
        # pass through the first hidden layer + activation (relu)
        x = self.relu(self.layer_1(x))
        
        # pass through the second hidden layer + activation (relu)
        x = self.relu(self.layer_2(x))
        
        # pass through the third hidden layer + activation (relu)
        x = self.relu(self.layer_3(x))
        
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