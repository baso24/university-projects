import heapq
import itertools

from environment import (
    GOAL_STATE_8, GRID_SIZE_8, TARGET_POS_8,
    GOAL_STATE_15, GRID_SIZE_15, TARGET_POS_15,
    print_board, get_neighbors, generate_solvable_state
)

# Comment or uncomment to switch configuration

# 8-Puzzle Configuration

# GOAL_STATE = GOAL_STATE_8
# GRID_SIZE = GRID_SIZE_8
# TARGET_POS = TARGET_POS_8
# PUZZLE_NAME = "8-PUZZLE"

# 15-Puzzle Configuration

GOAL_STATE = GOAL_STATE_15
GRID_SIZE = GRID_SIZE_15
TARGET_POS = TARGET_POS_15
PUZZLE_NAME = "15-PUZZLE"

# returns the Manhattan distance heuristic for a given state
def manhattan_distance(state):
    dist = 0
    for i, val in enumerate(state):
        if val != 0:
            curr_x, curr_y = i % GRID_SIZE, i // GRID_SIZE
            targ_x, targ_y = TARGET_POS[val]
            dist += abs(curr_x - targ_x) + abs(curr_y - targ_y)
    return dist

# implementation of a*
# inputs: initial state and heuristic function
# returns: path, number of nodes expanded, optimal cost
def a_star(start_state, heuristic_fn, weight_factor=1, max_nodes=None):
    # priority queue: (f_score = g_score + h_score, counter, state)
    # counter is used to break ties between states with the same f_score
    # g_score is the cost from the start state to the current state
    # h_score is the heuristic estimate of the cost from the current state to the goal state
    # f_score is the estimated total cost from the start state to the goal state through the current state
    open_list = []
    tie_breaker = itertools.count()
    
    # initialization: calculate the initial heuristic using the passed function
    initial_h = heuristic_fn(start_state)
    # we push the initial state in the priority queue (f_score, counter, state)
    heapq.heappush(open_list, (initial_h, next(tie_breaker), start_state))
    
    # came_from: keep track of the path (parent, move)
    came_from = {start_state: None}
    # g_score: keep track of the cost from the start state to the current state
    g_score = {start_state: 0}
    
    nodes_expanded = 0
    
    while open_list:
        if max_nodes is not None and nodes_expanded >= max_nodes:
            return None, nodes_expanded, -1
            
        _, _, current = heapq.heappop(open_list)
        
        # check if we have reached the goal
        if current == GOAL_STATE:
            path = []
            curr_trace = current
            while came_from[curr_trace] is not None:
                parent, move = came_from[curr_trace]
                path.append((move, curr_trace))
                curr_trace = parent
            path.reverse()
            return path, nodes_expanded, g_score[current]
        
        nodes_expanded += 1
        
        for neighbor, move_name in get_neighbors(current, GRID_SIZE):
            # tentative g_score
            tentative_g_score = g_score[current] + 1
            
            # if the neighbor is not in g_score or the tentative g_score is less than the current g_score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # update g_score
                g_score[neighbor] = tentative_g_score
                # compute the new f_score using the heuristic function
                h_score = heuristic_fn(neighbor)
                f_score = tentative_g_score + (weight_factor * h_score)
                # update came_from
                came_from[neighbor] = (current, move_name)
                heapq.heappush(open_list, (f_score, next(tie_breaker), neighbor))
                
    return None, nodes_expanded, -1

if __name__ == "__main__":
    print(f"--- {PUZZLE_NAME}: GENERATION AND RESOLUTION ---\n")
    
    # generate a solvable initial state
    initial_board = generate_solvable_state(GRID_SIZE)
    print("Generated Initial Configuration:")
    print_board(initial_board, GRID_SIZE)
    print(f"Initial Manhattan Heuristic: {manhattan_distance(initial_board)}")
    print("-" * 40)
    
    # execute A* passing the Manhattan distance as heuristic
    print("Starting A* guided by Manhattan Distance...\n")
    path, expanded, optimal_cost = a_star(initial_board, heuristic_fn=manhattan_distance)
    
    # debug and print
    if path is not None:
        print(f"Solution found! Expanded {expanded} nodes.\n")
        print("Move trace:")
        for step, (move, state) in enumerate(path, 1):
            print(f"Step {step}: Move '{move}'")
            print_board(state, GRID_SIZE)
            
        print("=" * 50)
        print(" DATASET ENTRY - READY FOR EXPORT ")
        print("=" * 50)
        print(f"[*] INPUT (Initial Configuration) : {initial_board}")
        print(f"[*] TARGET (Optimal Total Cost)    : {optimal_cost}")
        print("=" * 50)
    else:
        print("Error: No solution found (mathematically illegal).")