# davi_utils.py
# Small helpers shared by the DAVI training and evaluation scripts (8-puzzle only).

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import get_neighbors, GOAL_STATE_8, GRID_SIZE_8


# turn a board into network input
# Same encoding as dataset.py: for each tile we store its (row, col), ordered by
# tile value -> [row_0, col_0, row_1, col_1, ...]. dataset.py reads boards from a
# CSV; here we need to do it for boards we make on the fly during search/training.
def encode_state(board_state):
    num_tiles = GRID_SIZE_8 * GRID_SIZE_8
    coords = torch.zeros(num_tiles * 2, dtype=torch.float32)
    for pos_idx, tile_val in enumerate(board_state):
        coords[tile_val * 2] = pos_idx // GRID_SIZE_8      # row
        coords[tile_val * 2 + 1] = pos_idx % GRID_SIZE_8   # col
    return coords


def encode_states(states):
    # encode a list of boards into one batch tensor
    return torch.stack([encode_state(s) for s in states])


# make training boards without a solver
# Start from the solved board and take random moves backwards. Because every
# move can be undone, the result is always solvable and is at most num_moves
# away from the goal.
def scramble_from_goal(num_moves):
    state = GOAL_STATE_8
    prev = None
    for _ in range(num_moves):
        neighbors = [ns for ns, _ in get_neighbors(state, GRID_SIZE_8)]
        # try not to walk straight back to where we came from
        choices = [ns for ns in neighbors if ns != prev] or neighbors
        prev = state
        state = random.choice(choices)
    return state


# use the trained network as a heuristic for A*
class NeuralHeuristic:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = {}          # remember values so we don't recompute them
        self.model.eval()

    def __call__(self, state):
        if state == GOAL_STATE_8:
            return 0.0
        if state in self.cache:
            return self.cache[state]
        with torch.no_grad():
            x = encode_state(state).unsqueeze(0).to(self.device)
            value = self.model(x).item()
        value = max(0.0, value)   # moves-remaining can't be negative
        self.cache[state] = value
        return value


# the DAVI training target: J(s) = 0 if s is the goal, else 1 + min over neighbors J(s')
# target_net is a frozen copy of the network, used to supply J(s') for stability.
def compute_bellman_targets(states, target_net, device):
    # gather every neighbor of every non-goal board and run the network on them all at once
    all_neighbors = []
    neighbor_counts = []
    for s in states:
        if s == GOAL_STATE_8:
            neighbor_counts.append(0)
        else:
            nbrs = [ns for ns, _ in get_neighbors(s, GRID_SIZE_8)]
            neighbor_counts.append(len(nbrs))
            all_neighbors.extend(nbrs)

    with torch.no_grad():
        if all_neighbors:
            neighbor_values = target_net(encode_states(all_neighbors).to(device)).squeeze(1)
            neighbor_values = torch.clamp(neighbor_values, min=0.0)   # moves-remaining can't be negative
        else:
            neighbor_values = torch.empty(0, device=device)

    targets = torch.zeros(len(states), device=device)
    idx = 0
    for i, count in enumerate(neighbor_counts):
        if count > 0:
            targets[i] = (1.0 + neighbor_values[idx:idx + count]).min()
            idx += count
    return targets


# save a training-loss curve; returns False (without raising) if matplotlib isn't installed
def save_loss_plot(loss_history, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("loss (MSE)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(path)
    return True
