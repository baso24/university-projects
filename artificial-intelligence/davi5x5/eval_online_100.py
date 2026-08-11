import os
import sys
import time
import json
import heapq
import itertools
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from davi_model import DAVI_Ultra
from davi_utils import GOAL_STATE, get_neighbors_fast, NeuralHeuristic, ManhattanHeuristic

def a_star_5x5(start_state, heuristic_fn, max_nodes=None):
    if hasattr(heuristic_fn, 'evaluate_single'):
        h_func = heuristic_fn.evaluate_single
        h_batch_func = getattr(heuristic_fn, 'evaluate_batch', None)
    else:
        h_func = heuristic_fn
        h_batch_func = None

    start_tuple = tuple(start_state)
    open_list = []
    tie_breaker = itertools.count()
    
    initial_h = h_func(start_tuple)
    heapq.heappush(open_list, (initial_h, next(tie_breaker), start_tuple))
    
    came_from = {start_tuple: None}
    g_score = {start_tuple: 0}
    nodes_expanded = 0
    
    while open_list:
        if max_nodes is not None and nodes_expanded >= max_nodes:
            return None, nodes_expanded, -1
            
        _, _, current = heapq.heappop(open_list)
        
        if current == GOAL_STATE:
            path = []
            curr_trace = current
            while came_from[curr_trace] is not None:
                curr_trace = came_from[curr_trace]
                path.append(curr_trace)
            path.reverse()
            return path, nodes_expanded, g_score[current]
            
        nodes_expanded += 1
        
        neighbors = get_neighbors_fast(current)
        candidates = []
        for neighbor in neighbors:
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                candidates.append((neighbor, tentative_g))
                
        if candidates:
            cand_states = [c[0] for c in candidates]
            if h_batch_func:
                h_vals = h_batch_func(cand_states)
            else:
                h_vals = [h_func(s) for s in cand_states]
                
            for (neighbor, tentative_g), h_score in zip(candidates, h_vals):
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h_score
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_score, next(tie_breaker), neighbor))
                
    return None, nodes_expanded, -1

def evaluate_online():
    # Hardware detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running Online Evaluation on: {device}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "davi_model_5x5.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(current_directory, "davi_model.pth")
    dataset_path = os.path.join(current_directory, "test_dataset_100.json")

    # Model Loading
    model = DAVI_Ultra().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Unable to find '{model_path}'. Make sure training is completed.")
        return
    
    model.eval()
    neural_h = NeuralHeuristic(model, device)
    manhattan_h = ManhattanHeuristic()

    # Benchmark Dataset Loading
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Unable to find dataset '{dataset_path}'. Run the generator first.")
        return

    # Grouping boards by scramble_depth and CAST TO TUPLE (Resolves unhashable type)
    problems_by_depth = {}
    for item in dataset:
        d = item["scramble_depth"]
        if d not in problems_by_depth:
            problems_by_depth[d] = []
        problems_by_depth[d].append(tuple(item["board"]))

    print(f"Dataset loaded: {len(dataset)} total configurations.\n")

    # Table header
    print("=" * 135)
    print(f"{'Depth':<6} | {'BASELINE: A* + MANHATTAN (Averages)':<45} | {'OURS: A* + NEURAL HEURISTIC (Averages)':<48} | {'Advantage'}")
    print(f"{'':<6} | {'Exp. Nodes':<12} {'Time (s)':<12} {'Path Len':<19} | {'Exp. Nodes':<12} {'Time (s)':<12} {'Path Len':<12} {'% Optimal':<9} | {'Node Reduc.'}")
    print("-" * 135)

    global_nodes_m = 0
    global_nodes_n = 0

    # Iterate through each sorted difficulty level
    for depth in sorted(problems_by_depth.keys()):
        boards = problems_by_depth[depth]
        num_boards = len(boards)
        
        # Accumulators for block statistics
        sum_expanded_m, sum_time_m, sum_len_m = 0, 0, 0
        sum_expanded_n, sum_time_n, sum_len_n = 0, 0, 0
        optimal_count = 0
        timeout_m_count = 0

        sys.stdout.write(f"\rEvaluating batch Depth {depth} in progress ({num_boards} boards)... ")
        sys.stdout.flush()

        for state in boards:
            # 1. BASELINE: Manhattan (with safety limit of 500k nodes to prevent OOM)
            start_m = time.perf_counter()
            path_m, exp_m, _ = a_star_5x5(state, manhattan_h, max_nodes=500000)
            time_m = time.perf_counter() - start_m
            
            # 2. OURS: Neural Heuristic (No limit)
            start_n = time.perf_counter()
            path_n, exp_n, _ = a_star_5x5(state, neural_h)
            time_n = time.perf_counter() - start_n
            
            len_n = len(path_n) if path_n is not None else 0
            sum_expanded_n += exp_n
            sum_time_n += time_n
            sum_len_n += len_n
            
            # Statistics and Timeout Handling
            if path_m is not None:
                # Manhattan solved within 500k nodes
                len_m = len(path_m)
                sum_expanded_m += exp_m
                sum_time_m += time_m
                sum_len_m += len_m
                
                # Neural optimality check only if we have the true optimal path from the baseline
                if len_n == len_m:
                    optimal_count += 1
            else:
                # TIMEOUT: Manhattan did not find the solution in time
                timeout_m_count += 1
                sum_expanded_m += exp_m # Add the 500,000 wasted nodes
                sum_time_m += time_m    # Add elapsed time
                sum_len_m += len_n      # Use the neural len so as not to drop the visual average of Path Len

        # Average Calculation
        avg_exp_m = sum_expanded_m / num_boards
        avg_time_m = sum_time_m / num_boards
        avg_len_m = sum_len_m / num_boards

        avg_exp_n = sum_expanded_n / num_boards
        avg_time_n = sum_time_n / num_boards
        avg_len_n = sum_len_n / num_boards
        
        # Calculate % optimal excluding timeouts from the denominator
        solved_by_m = num_boards - timeout_m_count
        perc_optimal = (optimal_count / solved_by_m * 100) if solved_by_m > 0 else 100.0
        
        node_reduction = avg_exp_m / avg_exp_n if avg_exp_n > 0 else 0
        
        global_nodes_m += sum_expanded_m
        global_nodes_n += sum_expanded_n

        # Progress line cleanup
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        
        # Formatting Manhattan Path Len string to highlight any Timeouts
        timeout_str = f"(T: {timeout_m_count})" if timeout_m_count > 0 else ""
        len_m_str = f"{avg_len_m:.2f} {timeout_str}"
        
        print(f"{depth:<6} | {avg_exp_m:<12.1f} {avg_time_m:<12.3f} {len_m_str:<19} | {avg_exp_n:<12.1f} {avg_time_n:<12.3f} {avg_len_n:<12.2f} {perc_optimal:>8.1f}% | {node_reduction:>8.1f}x")

    print("=" * 135)
    
    if global_nodes_n > 0:
        tot_reduction = global_nodes_m / global_nodes_n
        print(f"\n🔥 Average global reduction of expanded nodes across the entire dataset: {tot_reduction:.2f}x\n")

if __name__ == "__main__":
    evaluate_online()
