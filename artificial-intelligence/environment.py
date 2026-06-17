import random

# --- 8-PUZZLE CONFIGURATION ---
GOAL_STATE_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GRID_SIZE_8 = 3
# mapping between the values and their target positions
TARGET_POS_8 = {val: (i % GRID_SIZE_8, i // GRID_SIZE_8) for i, val in enumerate(GOAL_STATE_8)}

# --- 15-PUZZLE CONFIGURATION ---
GOAL_STATE_15 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
GRID_SIZE_15 = 4
# mapping between the values and their target positions
TARGET_POS_15 = {val: (i % GRID_SIZE_15, i // GRID_SIZE_15) for i, val in enumerate(GOAL_STATE_15)}

# print board as readable grid
def print_board(state, grid_size):
    for i in range(0, len(state), grid_size):
        row = state[i:i+grid_size]
        print(" ".join(f"{x:2}" if x != 0 else " _" for x in row))
    print()

# calculate the number of inversions to check if the initial state is solvable
# returns the number of inversions
def get_inversions(state):
    state_no_zero = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(state_no_zero)):
        for j in range(i + 1, len(state_no_zero)):
            if state_no_zero[i] > state_no_zero[j]:
                inversions += 1
    return inversions

# check if the given initial state is solvable
# returns True if it's solvable, False otherwise
def is_solvable(state, grid_size):
    inversions = get_inversions(state)
    # for 8-puzzle (odd grid size), the state is solvable if the number of inversions is even
    if grid_size % 2 != 0:
        return inversions % 2 == 0
    # for 15-puzzle (even grid size), the situation is different
    # We first have to check the position of the blank tile
    else:
        blank_idx = state.index(0)
        blank_row_from_bottom = grid_size - (blank_idx // grid_size)
        # if the row of the blank tile from the bottom is even, the number of inversions must be odd
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 != 0
        # if the row of the blank tile from the bottom is odd, the number of inversions must be even
        else:
            return inversions % 2 == 0

# generate a random initial configuration ensuring it is solvable
# returns the initial state as a tuple
def generate_solvable_state(grid_size):
    num_tiles = grid_size * grid_size
    state_list = list(range(num_tiles))
    random.shuffle(state_list)
    
    # if it's not solvable we just need to swap two non-zero tiles to make it solvable
    if not is_solvable(state_list, grid_size):
        idx1, idx2 = 0, 1
        if state_list[idx1] == 0: idx1 = 2
        if state_list[idx2] == 0: idx2 = 2
        state_list[idx1], state_list[idx2] = state_list[idx2], state_list[idx1]
        
    return tuple(state_list)

# generate the neighbors of a given state by moving the empty tile
# returns a list of tuples (neighbor_state, move_name)
def get_neighbors(state, grid_size):
    neighbors = []
    # find the index of the empty tile
    zero_idx = state.index(0)
    # calculate the coordinates of the empty tile
    zero_x, zero_y = zero_idx % grid_size, zero_idx // grid_size
    
    # possible moves: Up, Down, Left, Right
    moves = {
        'Up': (0, -1),
        'Down': (0, 1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }
    
    for move_name, (dx, dy) in moves.items():
        # calculate the new coordinates of the empty tile
        new_x, new_y = zero_x + dx, zero_y + dy
        # check if the new coordinates are within the grid boundaries
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            # calculate the index of the new empty tile
            new_idx = new_y * grid_size + new_x
            # create the new state by swapping the empty tile with the adjacent tile
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            # append the new state and the move name to the list of neighbors
            neighbors.append((tuple(new_state), move_name))
            
    return neighbors
