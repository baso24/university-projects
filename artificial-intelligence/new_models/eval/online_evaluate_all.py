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
from models.small.small_model import SmallPuzzleNet
from models.delta.delta_model import DeltaPuzzleNet
from models.multitask.multitask_model import MultiTaskDeltaPuzzleNet
from models.embedding.embedding_model import EmbeddingPuzzleNet
from environment import GRID_SIZE_8, TARGET_POS_8
from search import a_star, manhattan_distance

def get_positional_heuristic(board_state, model, device):
    coords = torch.zeros((9, 2), dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        coords[tile_val, 0] = pos_idx // GRID_SIZE_8
        coords[tile_val, 1] = pos_idx % GRID_SIZE_8
    return model(coords.flatten().unsqueeze(0).to(device)).item()

def get_delta_heuristic(board_state, model, device):
    deltas = torch.zeros((9, 2), dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        curr_x = pos_idx % GRID_SIZE_8
        curr_y = pos_idx // GRID_SIZE_8
        targ_x, targ_y = TARGET_POS_8[tile_val]
        deltas[tile_val, 0] = float(curr_x - targ_x)
        deltas[tile_val, 1] = float(curr_y - targ_y)
    return model(deltas.flatten().unsqueeze(0).to(device)).item()

def get_embedding_heuristic(board_state, model, device):
    state_tensor = torch.tensor([board_state], dtype=torch.long).to(device)
    return model(state_tensor).item()

def load_test_boards_from_csv(filename="test_dataset.csv", num_boards=10):
    file_path = os.path.join(root_dir, "data", "datasets", filename)
    test_boards = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            first_line = next(reader)
            if "board_state" not in first_line[0]:
                test_boards.append(([int(x) for x in first_line[0].split(',')], float(first_line[1])))
            for _ in range(num_boards - len(test_boards)):
                row = next(reader)
                test_boards.append(([int(x) for x in row[0].split(',')], float(row[1])))
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return test_boards

def evaluate_embedding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding Benchmark starting on: {device}\n")

    pos_model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    pos_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth"), map_location=device, weights_only=True))
    pos_model.eval()

    small_model = SmallPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    small_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "small", "best_small_puzzle_model.pth"), map_location=device, weights_only=True))
    small_model.eval()

    delta_model = DeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    delta_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "delta", "best_delta_model.pth"), map_location=device, weights_only=True))
    delta_model.eval()

    mt_model = MultiTaskDeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    mt_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "multitask", "best_multitask_model.pth"), map_location=device, weights_only=True))
    mt_model.eval()

    emb_model = EmbeddingPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    emb_path = os.path.join(root_dir, "models", "embedding", "best_embedding_model.pth")
    try:
        emb_model.load_state_dict(torch.load(emb_path, map_location=device, weights_only=True))
        emb_model.eval()
        print(f"[*] Embedding Model loaded from {os.path.basename(emb_path)}")
    except FileNotFoundError:
        print("Error: Could not find Embedding model weights. Run train_embedding.py first.")
        return

    test_data = load_test_boards_from_csv(num_boards=1000)
    
    print("\n" + "=" * 145)
    print("      BENCHMARK: CLASSIC vs POSITIONAL vs SMALL POSITIONAL vs DELTA vs MULTI-TASK vs EMBEDDING")
    print("=" * 145)

    times_c, times_p, times_s, times_d, times_mt, times_emb = [], [], [], [], [], []
    nodes_c, nodes_p, nodes_s, nodes_d, nodes_mt, nodes_emb = [], [], [], [], [], []

    with torch.no_grad():
        for i, (start_board, target_cost) in enumerate(test_data):
            # Classic
            t0 = time.perf_counter(); _, c_nodes, c_cost = a_star(tuple(start_board), manhattan_distance, weight_factor=1); t_c = time.perf_counter() - t0
            # Positional
            t0 = time.perf_counter(); _, p_nodes, p_cost = a_star(tuple(start_board), lambda b: get_positional_heuristic(b, pos_model, device), weight_factor=3); t_p = time.perf_counter() - t0
            # Small Positional
            t0 = time.perf_counter(); _, s_nodes, s_cost = a_star(tuple(start_board), lambda b: get_positional_heuristic(b, small_model, device), weight_factor=3); t_s = time.perf_counter() - t0
            # Delta
            t0 = time.perf_counter(); _, d_nodes, d_cost = a_star(tuple(start_board), lambda b: get_delta_heuristic(b, delta_model, device), weight_factor=3); t_d = time.perf_counter() - t0
            # Multi-Task Delta
            t0 = time.perf_counter(); _, mt_nodes, mt_cost = a_star(tuple(start_board), lambda b: get_delta_heuristic(b, mt_model, device), weight_factor=3); t_mt = time.perf_counter() - t0
            # Tile Embedding
            t0 = time.perf_counter(); _, emb_nodes, emb_cost = a_star(tuple(start_board), lambda b: get_embedding_heuristic(b, emb_model, device), weight_factor=3); t_emb = time.perf_counter() - t0

            times_c.append(t_c); times_p.append(t_p); times_s.append(t_s); times_d.append(t_d); times_mt.append(t_mt); times_emb.append(t_emb)
            nodes_c.append(c_nodes); nodes_p.append(p_nodes); nodes_s.append(s_nodes); nodes_d.append(d_nodes); nodes_mt.append(mt_nodes); nodes_emb.append(emb_nodes)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} boards...")

    print(f"\n{'Metric':<32} | {'Classic A*':<14} | {'Positional':<14} | {'Small Posit.':<14} | {'Delta':<14} | {'Multi-Task':<14} | {'Embedding':<14}")
    print("-" * 145)
    print(f"{'Average Nodes Expanded':<32} | {sum(nodes_c)/len(nodes_c):<14.1f} | {sum(nodes_p)/len(nodes_p):<14.1f} | {sum(nodes_s)/len(nodes_s):<14.1f} | {sum(nodes_d)/len(nodes_d):<14.1f} | {sum(nodes_mt)/len(nodes_mt):<14.1f} | {sum(nodes_emb)/len(nodes_emb):<14.1f}")
    print(f"{'Average Execution Time (s)':<32} | {sum(times_c)/len(times_c):<14.4f} | {sum(times_p)/len(times_p):<14.4f} | {sum(times_s)/len(times_s):<14.4f} | {sum(times_d)/len(times_d):<14.4f} | {sum(times_mt)/len(times_mt):<14.4f} | {sum(times_emb)/len(times_emb):<14.4f}")
    print("=" * 145 + "\n")

if __name__ == "__main__":
    evaluate_embedding()
