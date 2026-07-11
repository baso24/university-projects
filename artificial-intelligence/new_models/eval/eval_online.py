import os
import sys
import time
import torch
import csv

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

from models.positional.model import PuzzleNet
from environment import GRID_SIZE_8
from search import a_star, manhattan_distance

def get_neural_heuristic(board_state, model, device):
    # 9x2 tensor for coordinates
    coords = torch.zeros((9, 2), dtype=torch.float32)
    
    for pos_idx, tile_val in enumerate(board_state):
        row = pos_idx // GRID_SIZE_8
        col = pos_idx % GRID_SIZE_8
        coords[tile_val, 0] = row
        coords[tile_val, 1] = col
        
    # flatten, add batch dimension, move to device
    encoded_state = coords.flatten().unsqueeze(0).to(device)
    
    # fast inference
    cost_estimation = model(encoded_state).item()
    return cost_estimation

def load_test_boards_from_csv(filename="test_dataset.csv", num_boards=10):
    file_path = os.path.join(root_dir, "data", "datasets", filename)
    
    test_boards = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            # skip header
            first_line = next(reader)
            if "board_state" in first_line[0]:
                pass
            else:
                # if no header, process the first line
                board_str, cost_str = first_line
                board = [int(x) for x in board_str.split(',')]
                test_boards.append((board, float(cost_str)))
                
            for _ in range(num_boards - len(test_boards)):
                row = next(reader)
                board_str, cost_str = row
                board = [int(x) for x in board_str.split(',')]
                test_boards.append((board, float(cost_str)))
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
    return test_boards

def evaluate_online():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Online evaluation starting on: {device}\n")

    # load trained model
    model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    model_path = os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Neural Network weights loaded successfully from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find model weights at {model_path}.")
        return

    # extract test boards directly from the CSV
    num_tests = 10
    test_data = load_test_boards_from_csv(num_boards=num_tests)
    
    if not test_data:
        print("Error: Could not find test boards. Run data/generate_data.py first.")
        return

    print("\n" + "=" * 90)
    print("                CLASSIC A* vs NEURAL A*")
    print("=" * 90)

    # accumulators for final summary
    total_classic_time = []
    total_neural_time = []
    total_classic_nodes = []
    total_neural_nodes = []

    # disable gradient computation globally for the entire A* search
    with torch.no_grad():
        for i, (start_board, target_cost) in enumerate(test_data):
            print(f"\n--- Test Board #{i+1} ---")
            print(f"Initial Configuration Array: {start_board}")
            print(f"Mathematical Optimal Cost:   {int(target_cost)} moves\n")
            
            # using manhattan distance as heuristic
            t0_classic = time.perf_counter()
            
            classic_path, classic_nodes, classic_cost = a_star(
                start_state=tuple(start_board), 
                heuristic_fn=manhattan_distance,
                weight_factor=1
            )
            
            t1_classic = time.perf_counter()
            classic_time = t1_classic - t0_classic

            # using neural network as heuristic - 
            # using lambda function to inject the neural logic smoothly into A*
            neural_h_func = lambda board: get_neural_heuristic(board, model, device)
            t0_neural = time.perf_counter()

            neural_path, neural_nodes, neural_cost = a_star(
                start_state=tuple(start_board), 
                heuristic_fn=neural_h_func,
                weight_factor=3
            )

            t1_neural = time.perf_counter()
            neural_time = t1_neural - t0_neural

            total_classic_time.append(classic_time)
            total_neural_time.append(neural_time)
            total_classic_nodes.append(classic_nodes)
            total_neural_nodes.append(neural_nodes)

            # results
            print(f"{'Metric':<28} | {'Classic A* (Manhattan)':<28} | {'Neural A* ':<28}")
            print("-" * 90)
            
            # sub-optimality check
            classic_cost_str = f"{classic_cost} (Optimal)"
            neural_cost_str = f"{neural_cost}"
            if neural_cost > classic_cost:
                neural_cost_str += f" (+{neural_cost - classic_cost} moves)"
            else:
                neural_cost_str += " (Optimal)"
                
            print(f"{'Path Length Generated':<28} | {classic_cost_str:<28} | {neural_cost_str:<28}")
            
            # efficiency check
            print(f"{'Nodes Expanded':<28} | {classic_nodes:<28} | {neural_nodes:<28}")
            
            # calculate search space reduction percentage
            if classic_nodes > neural_nodes:
                reduction = ((classic_nodes - neural_nodes) / classic_nodes) * 100
                print(f"{'Search Space Reduction':<28} | {'-':<28} | {reduction:.1f}% fewer nodes")
            elif classic_nodes < neural_nodes:
                reduction = ((neural_nodes - classic_nodes) / classic_nodes) * 100
                print(f"{'Search Space Reduction':<28} | {'-':<28} | {reduction:.1f}% more nodes")
            else:
                print(f"{'Search Space Reduction':<28} | {'-':<28} | 0.0% difference")
                
            # time check
            print(f"{'Execution Time (Seconds)':<28} | {classic_time:<28.4f} | {neural_time:<28.4f}")

    # compute overall metrics
    avg_classic_time = sum(total_classic_time) / len(total_classic_time)
    avg_neural_time = sum(total_neural_time) / len(total_neural_time)
    avg_classic_nodes = sum(total_classic_nodes) / len(total_classic_nodes)
    avg_neural_nodes = sum(total_neural_nodes) / len(total_neural_nodes)

    print("\n" + "=" * 90)
    print("                FINAL SUMMARY (AVERAGE ACROSS ALL TEST BOARDS)")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Classic A* (Manhattan)':<25} | {'Neural A*':<25}")
    print("-" * 90)
    print(f"{'Average Nodes Expanded':<35} | {avg_classic_nodes:<25.1f} | {avg_neural_nodes:<25.1f}")
    print(f"{'Average Execution Time (Seconds)':<35} | {avg_classic_time:<25.4f} | {avg_neural_time:<25.4f}")
    if avg_classic_nodes > avg_neural_nodes:
        overall_reduction = ((avg_classic_nodes - avg_neural_nodes) / avg_classic_nodes) * 100
        print(f"{'Overall Search Space Reduction':<35} | {'-':<25} | {overall_reduction:.1f}% fewer nodes")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    evaluate_online()