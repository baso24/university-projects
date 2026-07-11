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

from environment import GRID_SIZE_8, GRID_SIZE_15

class PuzzleDataset(Dataset):
    def __init__(self, csv_file, grid_size=GRID_SIZE_8):
        self.data = []
        self.num_classes = grid_size * grid_size
        self.grid_size = grid_size  # stored for coordinate calculation
        
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
        
        # init tensor to hold (row, col) for each tile from 0 to num_classes-1
        # for 8-puzzle, num_classes is 9, so dimension is (9, 2)
        # for 15-puzzle, num_classes is 16, so dimension is (16, 2)
        coords = torch.zeros((self.num_classes, 2), dtype=torch.float32)
        
        for pos_idx, tile_val in enumerate(board_state):
            # map the flat array index to 2D grid coordinates
            row = pos_idx // self.grid_size
            col = pos_idx % self.grid_size
            
            # assign the coordinates to the specific tile value
            coords[tile_val, 0] = row
            coords[tile_val, 1] = col
        
        # at this point coords is a matrix of dimension (num_classes,2) where each row is the coordinate of a tile
        # flatten the coords of dimension (num_classes,2) matrix into a 1D vector of 2*num_classes total elements

        # so we will have as input for the neural network
        # [row_0, col_0, row_1, col_1, ..., row_(N-1), col_(N-1)]
        # where row_0, col_0 are the coordinates of the tile 0, row_1, col_1 are the coordinates of the tile 1, and so on...
        encoded_state = coords.flatten()
        target_cost = torch.tensor([cost], dtype=torch.float32)
        
        return encoded_state, target_cost

def get_dataloaders(batch_size=1):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, "datasets")
    
    train_path = os.path.join(datasets_directory, "train_dataset.csv")
    val_path = os.path.join(datasets_directory, "val_dataset.csv")
    test_path = os.path.join(datasets_directory, "test_dataset.csv")
    
    train_dataset = PuzzleDataset(train_path, grid_size=GRID_SIZE_8)
    val_dataset = PuzzleDataset(val_path, grid_size=GRID_SIZE_8)
    test_dataset = PuzzleDataset(test_path, grid_size=GRID_SIZE_8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    try:
        train_loader, _, _ = get_dataloaders()
        sample_inputs, sample_targets = next(iter(train_loader))
        
        print(f"Input Batch Dimension:     {sample_inputs.shape}")
        print(f"Target Batch Dimension:    {sample_targets.shape}")
        print(f"Input Data Type:           {sample_inputs.dtype}")
        print("Pipeline completed successfully. The dataset is ready for the neural network.")
        
    except FileNotFoundError:
        print("Error: The CSV files were not found. Run 'generate_data.py' first.")