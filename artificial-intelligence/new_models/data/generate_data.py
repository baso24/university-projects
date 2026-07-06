import os
import sys

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

from environment import generate_solvable_state, GRID_SIZE_8, GRID_SIZE_15
from search import a_star, manhattan_distance

# helper function for saving dataset in csv format
def save_dataset(current_directory, entries, filename):
    datasets_dir = os.path.join(current_directory, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    filepath = os.path.join(datasets_dir, filename)
    with open(filepath, "w") as f:
        f.write("board_state,optimal_cost\n")
        for entry in entries:
            # Converting the tuple (1, 2, 0...) into a string "1,2,0..."
            board_str = ",".join(map(str, entry[0]))
            f.write(f"{board_str};{entry[1]}\n")

if __name__ == "__main__":
    print("Training, validation and test dataset generation")

    # dataset parameters
    TOTAL_SAMPLES = 10000
    TRAIN_FRAC = 0.8
    VAL_FRAC = 0.1
    # TEST_FRAC is the remaining 0.1
    
    dataset_entries = []
    seen_configurations = set()
    
    print(f"Generating {TOTAL_SAMPLES} unique configurations")

    while len(dataset_entries) < TOTAL_SAMPLES:
        
        initial_configuration = generate_solvable_state(GRID_SIZE_8) # here we decide if we want to generate 8-puzzle or 15-puzzle
        
        # We check if the configuration is novel
        if initial_configuration not in seen_configurations:
            seen_configurations.add(initial_configuration)
            
            # we resolve the puzzle
            path, nodes_expanded, optimal_cost = a_star(initial_configuration, manhattan_distance)
            
            # saving the pair (Input, Target)
            if path is not None:
                dataset_entries.append((initial_configuration, optimal_cost))

    # slicing
    train_idx = int(TOTAL_SAMPLES * TRAIN_FRAC)
    val_idx = train_idx + int(TOTAL_SAMPLES * VAL_FRAC)
    
    train_entries = dataset_entries[:train_idx]
    val_entries = dataset_entries[train_idx:val_idx]
    test_entries = dataset_entries[val_idx:]

    print("\nDATASET:\n")
    print(f"Training Set:   {len(train_entries)} samples")
    print(f"Validation Set: {len(val_entries)} samples")
    print(f"Test Set:       {len(test_entries)} samples")

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # saving on disk
    save_dataset(current_directory, train_entries, "train_dataset.csv")
    save_dataset(current_directory, val_entries, "val_dataset.csv")
    save_dataset(current_directory, test_entries, "test_dataset.csv")

    print("\nDataset generated and saved successfully in CSV format.")