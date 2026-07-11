# eval_online.py

# Online evaluation of the COMPLETE hybrid system: A* guided by the DAVI-trained
# neural heuristic, compared against classical A* + Manhattan distance.
#
# It measures what actually matters for a search heuristic:
#   * nodes expanded (search effort)
#   * wall-clock time
#   * solution length (to check both stay optimal)

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import generate_solvable_state, GRID_SIZE_8
from search import a_star, manhattan_distance
from davi_model import PuzzleResNet
from davi_utils import NeuralHeuristic

NUM_PROBLEMS = 100


def summarize(name, nodes, times, lengths, optimal_lengths):
    avg_nodes = sum(nodes) / len(nodes)
    avg_time = sum(times) / len(times)
    avg_len = sum(lengths) / len(lengths)
    num_optimal = sum(1 for l, o in zip(lengths, optimal_lengths) if l == o)
    pct_optimal = 100.0 * num_optimal / len(lengths)
    print(f"{name:<28} | {avg_nodes:>12.1f} | {avg_time*1000:>10.3f} | "
          f"{avg_len:>8.2f} | {pct_optimal:>7.1f}%")
    return avg_nodes


def evaluate_online():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Online evaluation will run on: {device}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "davi_model.pth")

    model = PuzzleResNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: could not find 'davi_model.pth'. Run train_davi.py first.")
        return
    model.eval()

    neural_h = NeuralHeuristic(model, device)

    # generate a fixed set of solvable problems
    problems = [generate_solvable_state(GRID_SIZE_8) for _ in range(NUM_PROBLEMS)]
    print(f"Evaluating on {NUM_PROBLEMS} random solvable 8-puzzles...\n")

    # classical baseline: A* + Manhattan (optimal) 
    base_nodes, base_times, base_lengths = [], [], []
    for state in problems:
        start = time.perf_counter()
        path, expanded, cost = a_star(state, manhattan_distance)
        base_times.append(time.perf_counter() - start)
        base_nodes.append(expanded)
        base_lengths.append(len(path))
    optimal_lengths = base_lengths  # A* + Manhattan is optimal

    #  learned heuristic + A* (still optimal, should expand fewer nodes)
    learned_nodes, learned_times, learned_lengths = [], [], []
    for state in problems:
        start = time.perf_counter()
        path, expanded, cost = a_star(state, neural_h)
        learned_times.append(time.perf_counter() - start)
        learned_nodes.append(expanded)
        learned_lengths.append(len(path))

    # report 
    print("=" * 82)
    print(f"{'METHOD':<28} | {'Avg Nodes':>12} | {'Avg ms':>10} | "
          f"{'Avg Len':>8} | {'Optimal':>8}")
    print("-" * 82)
    base_avg = summarize("A* + Manhattan", base_nodes, base_times,
                         base_lengths, optimal_lengths)
    learned_avg = summarize("A* + Learned heuristic", learned_nodes, learned_times,
                            learned_lengths, optimal_lengths)
    print("=" * 82)

    print(f"\nNode-expansion reduction vs classical A* + Manhattan: "
          f"{base_avg / max(learned_avg, 1e-9):.2f}x fewer nodes\n")


if __name__ == "__main__":
    evaluate_online()
