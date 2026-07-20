# train_davi.py
# Trains the heuristic with DAVI (Deep Approximate Value Iteration).
# Unlike train.py, this does NOT need a dataset of solved puzzles: it learns the
# cost-to-go J(s) straight from the Bellman rule
#     J(s) = 0                                  if s is the goal
#     J(s) = 1 + min over neighbors s' of J(s')  otherwise
# A frozen "target network" supplies J(s') and periodically catches up to the
# live network, which keeps the bootstrapped targets from chasing a moving
# goalpost (see davi_utils.compute_bellman_targets).


import os
import copy
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from davi_model import PuzzleResNet
from davi_utils import scramble_from_goal, encode_states, compute_bellman_targets, save_loss_plot

ITERATIONS = 10000 
BATCH_SIZE = 1000
LEARNING_RATE = 0.0005  # Slightly reduced for deeper networks
SYNC_EVERY = 100        # Increased to stabilize targets in the first phase

MODEL_FILE = "davi_model.pth"
PLOT_FILE = "davi_loss_curve.png"


def train_davi():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training DAVI for the 15-puzzle on {device}")

    model = PuzzleResNet().to(device)
    target_net = copy.deepcopy(model).to(device)   # frozen copy that supplies the targets
    target_net.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []

    current_max_scramble = 2
    iterations_in_level = 0
    
    for iteration in range(1, ITERATIONS + 1):
        # The number of iterations at the current level scales with the difficulty.
        # E.g.: max_scramble=2 -> 100 iterations
        # E.g.: max_scramble=20 -> 1000 iterations
        # We set a maximum (e.g., 2000 or 3000) to prevent the highest levels 
        # from taking too long to step up.
        required_iters = min(2000, current_max_scramble * 50)
        
        if iterations_in_level >= required_iters and current_max_scramble < 80:
            current_max_scramble += 1
            iterations_in_level = 0
            print(f"\n[Iter {iteration}] 🚀 Curriculum Step-Up: MAX_SCRAMBLE = {current_max_scramble}")
        
        iterations_in_level += 1

        # Generate batch with implicit experience replay (from 1 to current_max_scramble)
        states = [scramble_from_goal(random.randint(1, current_max_scramble)) for _ in range(BATCH_SIZE)]
        targets = compute_bellman_targets(states, target_net, device)

        model.train()
        preds = model(encode_states(states).to(device)).squeeze(1)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if iteration % SYNC_EVERY == 0:
            target_net.load_state_dict(model.state_dict())
            target_net.eval()
            print(f"iter {iteration}/{ITERATIONS}  loss={loss.item():.4f}")

    here = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(here, MODEL_FILE))
    print(f"Saved model to {MODEL_FILE}")

    if save_loss_plot(loss_history, os.path.join(here, PLOT_FILE), "DAVI 8-puzzle training loss"):
        print(f"Saved loss curve to {PLOT_FILE}")


if __name__ == "__main__":
    train_davi()
