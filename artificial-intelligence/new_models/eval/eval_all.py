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
from environment import GRID_SIZE_8
from search import a_star, manhattan_distance

def get_neural_heuristic(board_state, model, device):
    coords = torch.zeros((9, 2), dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        row = pos_idx // GRID_SIZE_8
        col = pos_idx % GRID_SIZE_8
        coords[tile_val, 0] = row
        coords[tile_val, 1] = col
    encoded_state = coords.flatten().unsqueeze(0).to(device)
    cost_estimation = model(encoded_state).item()
    return cost_estimation

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

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comprehensive Benchmark starting on: {device}\n")

    # Load Teacher Model
    teacher_model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    teacher_path = os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth")
    try:
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher_model.eval()
        print(f"[*] Teacher Model (PuzzleNet) loaded from {os.path.basename(teacher_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find teacher model weights at {teacher_path}.")
        return

    # Load Student Model
    student_model = SmallPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    student_path = os.path.join(root_dir, "models", "small", "best_small_puzzle_model.pth")
    try:
        student_model.load_state_dict(torch.load(student_path, map_location=device, weights_only=True))
        student_model.eval()
        print(f"[*] Student Model (SmallPuzzleNet) loaded from {os.path.basename(student_path)}")
    except FileNotFoundError:
        print("Error: Could not find student model weights.")
        return

    test_data = load_test_boards_from_csv(num_boards=10)
    if not test_data:
        return

    print("\n" + "=" * 105)
    print("                COMPREHENSIVE BENCHMARK: CLASSIC vs TEACHER vs STUDENT")
    print("=" * 105)

    times_classic, times_teacher, times_student = [], [], []
    nodes_classic, nodes_teacher, nodes_student = [], [], []

    with torch.no_grad():
        for i, (start_board, target_cost) in enumerate(test_data):
            print(f"\n--- Test Board #{i+1} (Optimal Cost: {int(target_cost)} moves) ---")
            
            # 1. Classic A*
            t0 = time.perf_counter()
            _, c_nodes, c_cost = a_star(tuple(start_board), manhattan_distance, weight_factor=1)
            t_classic = time.perf_counter() - t0

            # 2. Teacher A*
            t0 = time.perf_counter()
            _, t_nodes, t_cost = a_star(tuple(start_board), lambda b: get_neural_heuristic(b, teacher_model, device), weight_factor=3)
            t_teacher = time.perf_counter() - t0

            # 3. Student A*
            t0 = time.perf_counter()
            _, s_nodes, s_cost = a_star(tuple(start_board), lambda b: get_neural_heuristic(b, student_model, device), weight_factor=3)
            t_student = time.perf_counter() - t0

            times_classic.append(t_classic)
            times_teacher.append(t_teacher)
            times_student.append(t_student)
            nodes_classic.append(c_nodes)
            nodes_teacher.append(t_nodes)
            nodes_student.append(s_nodes)

            print(f"{'Metric':<25} | {'Classic A* (Manhattan)':<22} | {'Teacher A* (Large)':<22} | {'Student A* (Small)':<22}")
            print("-" * 105)
            print(f"{'Path Length':<25} | {c_cost:<22} | {t_cost:<22} | {s_cost:<22}")
            print(f"{'Nodes Expanded':<25} | {c_nodes:<22} | {t_nodes:<22} | {s_nodes:<22}")
            print(f"{'Execution Time (s)':<25} | {t_classic:<22.4f} | {t_teacher:<22.4f} | {t_student:<22.4f}")

    print("\n" + "=" * 105)
    print("                FINAL BENCHMARK SUMMARY (AVERAGE ACROSS ALL BOARDS)")
    print("=" * 105)
    print(f"{'Metric':<35} | {'Classic A*':<20} | {'Teacher (Large)':<20} | {'Student (Small)':<20}")
    print("-" * 105)
    print(f"{'Average Nodes Expanded':<35} | {sum(nodes_classic)/len(nodes_classic):<20.1f} | {sum(nodes_teacher)/len(nodes_teacher):<20.1f} | {sum(nodes_student)/len(nodes_student):<20.1f}")
    print(f"{'Average Execution Time (Seconds)':<35} | {sum(times_classic)/len(times_classic):<20.4f} | {sum(times_teacher)/len(times_teacher):<20.4f} | {sum(times_student)/len(times_student):<20.4f}")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    evaluate_all()
