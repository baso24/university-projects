import random

# Define the goal state for the 8-puzzle (3x3)
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GRID_SIZE = 3

# Pre-calculate the target (x, y) coordinates to optimize Manhattan distance
TARGET_POS = {val: (i % GRID_SIZE, i // GRID_SIZE) for i, val in enumerate(GOAL_STATE)}

def print_board(state):
    """Print the tuple as a readable 3x3 grid."""
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

def get_inversions(state):
    """Calculate the number of inversions for parity check."""
    state_no_zero = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(state_no_zero)):
        for j in range(i + 1, len(state_no_zero)):
            if state_no_zero[i] > state_no_zero[j]:
                inversions += 1
    return inversions

def generate_solvable_state():
    """Generate a random initial configuration ensuring it is solvable."""
    state_list = list(range(9))
    random.shuffle(state_list)
    
    # For a grid with an odd width (3x3), the puzzle is solvable 
    # if and only if the number of inversions is even.
    if get_inversions(state_list) % 2 != 0:
        # If it's odd, swap two non-zero tiles to invert the parity
        idx1, idx2 = 0, 1
        if state_list[idx1] == 0: idx1 = 2
        if state_list[idx2] == 0: idx2 = 2
        state_list[idx1], state_list[idx2] = state_list[idx2], state_list[idx1]
        
    return tuple(state_list)

def get_neighbors(state):
    """Generate legal child states by moving the empty tile (0)."""
    neighbors = []
    zero_idx = state.index(0)
    zero_x, zero_y = zero_idx % GRID_SIZE, zero_idx // GRID_SIZE
    
    # Possible moves (dx, dy): Up, Down, Left, Right
    moves = {
        'Up': (0, -1),
        'Down': (0, 1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }
    
    for move_name, (dx, dy) in moves.items():
        new_x, new_y = zero_x + dx, zero_y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            new_idx = new_y * GRID_SIZE + new_x
            
            # Create the new state by swapping the zero with the target tile
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move_name))
            
    return neighbors
