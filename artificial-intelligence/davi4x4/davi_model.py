# davi_model.py
# A deeper network for DAVI. It has the same job as model.py's PuzzleNet
# (turn a board into one number = estimated cost-to-go), but with a few extra
# layers and skip connections so it can learn the harder value-iteration target.

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import GRID_SIZE_15


class ResidualBlock(nn.Module):
    # two linear layers where we add the input back at the end (a "skip").
    # this makes deeper networks easier to train.
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.bn_1 = nn.BatchNorm1d(dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.bn_2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn_1(self.linear_1(x)))
        out = self.bn_2(self.linear_2(out))
        return self.relu(out + residual)   # add the skip, then activate


class PuzzleResNet(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4):
        super(PuzzleResNet, self).__init__()
        self.num_tiles = GRID_SIZE_15 * GRID_SIZE_15
        # same input as PuzzleNet: (row, col) for every tile -> 2 * num_tiles
        self.input_size = self.num_tiles * 2

        self.input_layer = nn.Linear(self.input_size, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        # one number out = estimated moves to the goal
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.input_bn(self.input_layer(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)
