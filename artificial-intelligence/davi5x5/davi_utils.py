import random
import torch
import torch.nn.functional as F

GRID_SIZE = 5
NUM_TILES = 25
GOAL_STATE = tuple(list(range(1, NUM_TILES)) + [0])

# Precomputed O(1) move lookup table
VALID_MOVES = {}
for i in range(NUM_TILES):
    r, c = divmod(i, GRID_SIZE)
    moves = []
    if r > 0: moves.append(-GRID_SIZE)       # Up
    if r < GRID_SIZE - 1: moves.append(GRID_SIZE) # Down
    if c > 0: moves.append(-1)               # Left
    if c < GRID_SIZE - 1: moves.append(1)    # Right
    VALID_MOVES[i] = moves

def get_neighbors_fast(state):
    """Generates valid neighbor board configurations."""
    idx = state.index(0)
    neighbors = []
    for m in VALID_MOVES[idx]:
        new_idx = idx + m
        new_state = list(state)
        new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[idx]
        neighbors.append(tuple(new_state))
    return neighbors

def generate_true_scramble(goal_state, scramble_steps):
    """Generates deep non-looping scrambles."""
    state = goal_state
    blank_idx = state.index(0)
    last_blank_idx = -1
    
    for _ in range(scramble_steps):
        valid = VALID_MOVES[blank_idx]
        choices = [m for m in valid if (blank_idx + m) != last_blank_idx]
        if not choices:
            choices = valid
            
        m = random.choice(choices)
        next_blank_idx = blank_idx + m
        
        new_state = list(state)
        new_state[blank_idx], new_state[next_blank_idx] = new_state[next_blank_idx], new_state[blank_idx]
        
        last_blank_idx = blank_idx
        blank_idx = next_blank_idx
        state = tuple(new_state)
        
    return state

def encode_states_fast(states, device):
    """Vectorized PyTorch One-Hot Encoding on GPU."""
    state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    encoded = F.one_hot(state_tensor, num_classes=NUM_TILES).float()
    return encoded.view(state_tensor.size(0), -1)

class NeuralHeuristic:
    """Wrapper for fast batch evaluation in A* search."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_batch(self, states):
        x = encode_states_fast(states, self.device)
        preds = self.model(x).squeeze(-1)
        return torch.clamp(preds, min=0.0).tolist()

    def evaluate_single(self, state):
        return self.evaluate_batch([state])[0]

class ManhattanHeuristic:
    """Classic 5x5 Manhattan distance baseline."""
    def evaluate_single(self, state):
        dist = 0
        for idx, tile in enumerate(state):
            if tile == 0: continue
            curr_r, curr_c = divmod(idx, GRID_SIZE)
            goal_r, goal_c = divmod(tile - 1, GRID_SIZE)
            dist += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return dist
        
    def evaluate_batch(self, states):
        return [self.evaluate_single(s) for s in states]