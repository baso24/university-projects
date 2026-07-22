# eval_online.py

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import a_star, manhattan_distance
from davi_model import PuzzleResNet
from davi_utils import NeuralHeuristic, scramble_from_goal

def evaluate_online():
    # Hardware detection (including MPS for Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running Online Evaluation on: {device}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "davi_model.pth")  # or davi_model_final.pth

    # Initialization of the ResNet configured for current training
    model = PuzzleResNet(hidden_dim=256, num_blocks=4).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Unable to find '{model_path}'. Make sure training is completed.")
        return
    
    model.eval()
    neural_h = NeuralHeuristic(model, device)

    # Generation of the test set with scramble steps up to 60
    scramble_depths = list(range(10, 60, 2))
    problems = [(depth, scramble_from_goal(depth)) for depth in scramble_depths]
    
    print(f"Evaluating on {len(problems)} states, generated with scramble from 0 to 60...\n")

    # Table header for console printing
    print("=" * 110)
    print(f"{'Depth':<6} | {'BASELINE: A* + MANHATTAN':<45} | {'OURS: A* + NEURAL HEURISTIC':<45}")
    print(f"{'':<6} | {'Exp. Nodes':<12} {'Time (s)':<12} {'Path Len':<19} | {'Exp. Nodes':<12} {'Time (s)':<12} {'Path Len':<19}")
    print("-" * 110)

    tot_nodes_manhattan = 0
    tot_nodes_neural = 0

    for depth, state in problems:
        # --- 1. A* + Manhattan (Optimal Baseline) ---
        start = time.perf_counter()
        
        # WARNING: for depth > 45, on 15-puzzle A* with Manhattan risks
        # expanding millions of nodes, saturating RAM. If it freezes, you might
        # need to set a timeout inside your a_star function.
        path_m, expanded_m, cost_m = a_star(state, manhattan_distance)
        
        time_m = time.perf_counter() - start
        len_m = len(path_m)
        tot_nodes_manhattan += expanded_m

        # --- 2. A* + Neural Heuristic ---
        start = time.perf_counter()
        path_n, expanded_n, cost_n = a_star(state, neural_h)
        time_n = time.perf_counter() - start
        len_n = len(path_n)
        tot_nodes_neural += expanded_n
        
        # Admissibility check: did the neural network find the optimal path?
        # If the network overestimates h(n), A* loses admissibility guarantees.
        opt_str_n = f"{len_n}" if len_n == len_m else f"{len_n} (Sub-optimal!)"

        # Print comparison row
        print(f"{depth:<6} | {expanded_m:<12} {time_m:<12.3f} {len_m:<19} | {expanded_n:<12} {time_n:<12.3f} {opt_str_n:<19}")

    # Aggregated Statistics
    print("=" * 110)
    if tot_nodes_neural > 0:
        reduction = tot_nodes_manhattan / tot_nodes_neural
        print(f"\n🔥 Overall expanded nodes reduction: {reduction:.2f}x in favor of the Neural Network.\n")

if __name__ == "__main__":
    evaluate_online()