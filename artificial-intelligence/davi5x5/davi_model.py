import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernResidualBlock(nn.Module):
    """Pre-LayerNorm Residual Block for stable deep gradient flow."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        res = x
        x = F.gelu(self.fc1(self.norm1(x)))
        x = self.fc2(self.norm2(x))
        return x + res

class DAVI_Ultra(nn.Module):
    """
    Deep Approximate Value Iteration Network for 5x5 (24-Puzzle).
    Input: Flat one-hot encoding of 25 tiles across 25 positions (625 dims).
    """
    def __init__(self, input_dim=625, hidden_dim=1024, num_blocks=6):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ModernResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out1 = nn.Linear(hidden_dim, 256)
        self.fc_out2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.fc_in(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.fc_out1(x))
        return self.fc_out2(x)