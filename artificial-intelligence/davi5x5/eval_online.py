import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
import torch

from davi_model import DAVI_Ultra
from davi_utils import (
    GOAL_STATE, get_neighbors_fast, generate_true_scramble,
    NeuralHeuristic, ManhattanHeuristic
)

def find_checkpoint():
    """Local checkpoint lookup: Checks the current directory for priority files or any .pth checkpoint."""
    priority_names = ["davi_model_5x5.pth"]
    for fname in priority_names:
        if os.path.exists(fname):
            return fname
            
    local_pth_files = [f for f in os.listdir(".") if f.endswith(".pth")]
    if local_pth_files:
        # Sort alphabetically/numerically to pick the latest iteration if multiple exist
        return sorted(local_pth_files)[-1]
        
    return None

def reconstruct_path(parent_map, goal):
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent_map[curr]
    return path[::-1]

def a_star_solve(start_state, heuristic_class, weight=1.0, max_nodes=100000):
    frontier = []
    h_start = heuristic_class.evaluate_single(start_state)
    heapq.heappush(frontier, (weight * h_start, 0, 0, start_state))
    
    g_score = {start_state: 0}
    parent_map = {start_state: None}
    nodes_expanded = 0
    state_counter = 1
    
    while frontier:
        if nodes_expanded >= max_nodes:
            return None, max_nodes, True
            
        f, current_g, _, current_state = heapq.heappop(frontier)
        
        if current_state == GOAL_STATE:
            return reconstruct_path(parent_map, current_state), nodes_expanded, False
            
        if current_g > g_score.get(current_state, float('inf')):
            continue
            
        nodes_expanded += 1
        
        all_neighbors = get_neighbors_fast(current_state)
        candidates = []
        for n in all_neighbors:
            tentative_g = current_g + 1
            if tentative_g < g_score.get(n, float('inf')):
                candidates.append((n, tentative_g))
                
        if candidates:
            cand_states = [c[0] for c in candidates]
            h_vals = heuristic_class.evaluate_batch(cand_states)
            
            for (neighbor, tentative_g), h in zip(candidates, h_vals):
                g_score[neighbor] = tentative_g
                parent_map[neighbor] = current_state
                f_new = tentative_g + (weight * h)
                heapq.heappush(frontier, (f_new, tentative_g, state_counter, neighbor))
                state_counter += 1
                
    return None, nodes_expanded, True

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Online Evaluation on: {device}")
    
    model = DAVI_Ultra().to(device)
    ckpt = find_checkpoint()
    
    if ckpt:
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print("Model checkpoint loaded successfully.")
    else:
        print("Error: Could not locate any .pth model checkpoint in current folder or Kaggle input.")
        return

    heuristics = {
        "Manhattan (W=1.0)": (ManhattanHeuristic(), 1.0),
        "Neural Net (W=1.0)": (NeuralHeuristic(model, device), 1.0),
        "Neural Net (W=1.3)": (NeuralHeuristic(model, device), 1.3)
    }
    
    test_depths = [20, 40, 60, 80, 100]
    num_tests = 10
    node_limit = 100000 
    
    results_nodes = {name: [] for name in heuristics}
    results_success = {name: [] for name in heuristics}
    results_path_len = {name: [] for name in heuristics}

    for depth in test_depths:
        print(f"\n--- Benchmark Depth: {depth} ---")
        test_boards = [generate_true_scramble(GOAL_STATE, depth) for _ in range(num_tests)]
        
        for name, (h_func, weight) in heuristics.items():
            nodes_list, path_lengths = [], []
            success_count = 0
            
            for board in test_boards:
                path, nodes, timed_out = a_star_solve(board, h_func, weight=weight, max_nodes=node_limit)
                nodes_list.append(nodes)
                if not timed_out:
                    success_count += 1
                    path_lengths.append(len(path) - 1)
            
            avg_nodes = np.mean(nodes_list)
            success_rate = (success_count / num_tests) * 100
            avg_path = np.mean(path_lengths) if path_lengths else 0
            
            results_nodes[name].append(avg_nodes)
            results_success[name].append(success_rate)
            results_path_len[name].append(avg_path)
            
            print(f"{name:<20} | Success: {success_rate:>5.1f}% | Avg Nodes: {avg_nodes:>8.1f}")

    # Generate Performance Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 5))
    for name, nodes in results_nodes.items():
        plt.plot(test_depths, nodes, marker='o', label=name, linewidth=2)
    plt.yscale('log')
    plt.title('Nodes Expanded Comparison (Log Scale)')
    plt.xlabel('Scramble Depth')
    plt.ylabel('Avg Expanded Nodes')
    plt.legend()
    plt.savefig('davi_loss_curve.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_benchmark()