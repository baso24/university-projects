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
from models.delta.delta_model import DeltaPuzzleNet
from environment import GRID_SIZE_8, TARGET_POS_8
from search import a_star, manhattan_distance

def get_positional_heuristic(board_state, model, device):
    coords = torch.zeros((9, 2), dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        row = pos_idx // GRID_SIZE_8
        col = pos_idx % GRID_SIZE_8
        coords[tile_val, 0] = row
        coords[tile_val, 1] = col
    encoded_state = coords.flatten().unsqueeze(0).to(device)
    return model(encoded_state).item()

def get_delta_heuristic(board_state, model, device):
    deltas = torch.zeros((9, 2), dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        curr_x = pos_idx % GRID_SIZE_8
        curr_y = pos_idx // GRID_SIZE_8
        targ_x, targ_y = TARGET_POS_8[tile_val]
        deltas[tile_val, 0] = float(curr_x - targ_x)
        deltas[tile_val, 1] = float(curr_y - targ_y)
    encoded_state = deltas.flatten().unsqueeze(0).to(device)
    return model(encoded_state).item()

def load_test_boards_from_csv(filename="test_dataset.csv", num_boards=10):
    file_path = os.path.join(root_dir, "data", "datasets", filename)
    test_boards = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            first_line = next(reader)
            if "board_state" not in first_line[0]:
                board_str, cost_str = first_line
                test_boards.append(([int(x) for x in board_str.split(',')], float(cost_str)))
            for _ in range(num_boards - len(test_boards)):
                row = next(reader)
                board_str, cost_str = row
                test_boards.append(([int(x) for x in board_str.split(',')], float(cost_str)))
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return test_boards

def evaluate_delta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Delta Benchmark starting on: {device}\n")

    # Load Positional Model
    pos_model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    pos_path = os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth")
    try:
        pos_model.load_state_dict(torch.load(pos_path, map_location=device, weights_only=True))
        pos_model.eval()
        print(f"[*] Positional Model (PuzzleNet) loaded from {os.path.basename(pos_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find positional model weights at {pos_path}.")
        return

    # Load Delta Model
    delta_model = DeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    delta_path = os.path.join(root_dir, "models", "delta", "best_delta_model.pth")
    try:
        delta_model.load_state_dict(torch.load(delta_path, map_location=device, weights_only=True))
        delta_model.eval()
        print(f"[*] Delta Model (DeltaPuzzleNet) loaded from {os.path.basename(delta_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find delta model weights at {delta_path}.")
        return

    test_data = load_test_boards_from_csv(num_boards=10)
    if not test_data:
        return

    print("\n" + "=" * 105)
    print("                BENCHMARK: CLASSIC vs POSITIONAL NEURAL vs DELTA NEURAL")
    print("=" * 105)

    times_classic, times_pos, times_delta = [], [], []
    nodes_classic, nodes_pos, nodes_delta = [], [], []

    with torch.no_grad():
        for i, (start_board, target_cost) in enumerate(test_data):
            print(f"\n--- Test Board #{i+1} (Optimal Cost: {int(target_cost)} moves) ---")
            
            # 1. Classic A*
            t0 = time.perf_counter()
            _, c_nodes, c_cost = a_star(tuple(start_board), manhattan_distance, weight_factor=1)
            t_classic = time.perf_counter() - t0

            # 2. Positional Neural A*
            t0 = time.perf_counter()
            _, p_nodes, p_cost = a_star(tuple(start_board), lambda b: get_positional_heuristic(b, pos_model, device), weight_factor=3)
            t_pos = time.perf_counter() - t0

            # 3. Delta Neural A*
            t0 = time.perf_counter()
            _, d_nodes, d_cost = a_star(tuple(start_board), lambda b: get_delta_heuristic(b, delta_model, device), weight_factor=3)
            t_delta = time.perf_counter() - t0

            times_classic.append(t_classic)
            times_pos.append(t_pos)
            times_delta.append(t_delta)
            nodes_classic.append(c_nodes)
            nodes_pos.append(p_nodes)
            nodes_delta.append(d_nodes)

            print(f"{'Metric':<25} | {'Classic A* (Manhattan)':<22} | {'Positional A*':<22} | {'Delta A* (dx,dy)':<22}")
            print("-" * 105)
            print(f"{'Path Length':<25} | {c_cost:<22} | {p_cost:<22} | {d_cost:<22}")
            print(f"{'Nodes Expanded':<25} | {c_nodes:<22} | {p_nodes:<22} | {d_nodes:<22}")
            print(f"{'Execution Time (s)':<25} | {t_classic:<22.4f} | {t_pos:<22.4f} | {t_delta:<22.4f}")

    print("\n" + "=" * 105)
    print("                FINAL BENCHMARK SUMMARY (AVERAGE ACROSS ALL BOARDS)")
    print("=" * 105)
    print(f"{'Metric':<35} | {'Classic A*':<20} | {'Positional A*':<20} | {'Delta A* (dx,dy)':<20}")
    print("-" * 105)
    print(f"{'Average Nodes Expanded':<35} | {sum(nodes_classic)/len(nodes_classic):<20.1f} | {sum(nodes_pos)/len(nodes_pos):<20.1f} | {sum(nodes_delta)/len(nodes_delta):<20.1f}")
    print(f"{'Average Execution Time (Seconds)':<35} | {sum(times_classic)/len(times_classic):<20.4f} | {sum(times_pos)/len(times_pos):<20.4f} | {sum(times_delta)/len(times_delta):<20.4f}")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    evaluate_delta()
