import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

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

from environment import GRID_SIZE_8, TARGET_POS_8

class DeltaPuzzleDataset(Dataset):
    def __init__(self, csv_file, grid_size=GRID_SIZE_8):
        self.data = []
        self.num_classes = grid_size * grid_size
        self.grid_size = grid_size
        self.target_pos = TARGET_POS_8 if grid_size == 3 else {}
        
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if "board_state" in lines[0]:
                lines = lines[1:]
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                board_str, cost_str = line.split(';')
                board_state = [int(x) for x in board_str.split(',')]
                cost = float(cost_str)
                self.data.append((board_state, cost))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, cost = self.data[idx]
        
        # Tensor to hold (dx, dy) for each tile from 0 to num_classes-1
        deltas = torch.zeros((self.num_classes, 2), dtype=torch.float32)
        
        for pos_idx, tile_val in enumerate(board_state):
            curr_x = pos_idx % self.grid_size
            curr_y = pos_idx // self.grid_size
            
            targ_x, targ_y = self.target_pos[tile_val]
            
            # Compute displacement delta (dx, dy)
            deltas[tile_val, 0] = float(curr_x - targ_x)
            deltas[tile_val, 1] = float(curr_y - targ_y)
        
        encoded_state = deltas.flatten()
        target_cost = torch.tensor([cost], dtype=torch.float32)
        return encoded_state, target_cost

def get_delta_dataloaders(batch_size=1):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, "datasets")
    
    train_path = os.path.join(datasets_directory, "train_dataset.csv")
    val_path = os.path.join(datasets_directory, "val_dataset.csv")
    test_path = os.path.join(datasets_directory, "test_dataset.csv")
    
    train_dataset = DeltaPuzzleDataset(train_path, grid_size=GRID_SIZE_8)
    val_dataset = DeltaPuzzleDataset(val_path, grid_size=GRID_SIZE_8)
    test_dataset = DeltaPuzzleDataset(test_path, grid_size=GRID_SIZE_8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, _, _ = get_delta_dataloaders()
    sample_inputs, sample_targets = next(iter(train_loader))
    print(f"Delta Input Batch Dimension:  {sample_inputs.shape}")
    print(f"Sample Delta Vector (tile 0 to 8): {sample_inputs[0].numpy()}")
