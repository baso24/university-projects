import heapq
import itertools

from environment import GOAL_STATE, GRID_SIZE, TARGET_POS
from environment import print_board, get_neighbors, generate_solvable_state

def manhattan_distance(state):
    """Calculate the Manhattan distance for the current state."""
    dist = 0
    for i, val in enumerate(state):
        if val != 0:
            curr_x, curr_y = i % GRID_SIZE, i // GRID_SIZE
            targ_x, targ_y = TARGET_POS[val]
            dist += abs(curr_x - targ_x) + abs(curr_y - targ_y)
    return dist


def a_star(start_state, heuristic_fn):
    """
    A* implementation. 
    Accepts the initial state and a heuristic function (e.g., Manhattan or Neural Network) as input.
    """
    # Priority queue: (f_score, counter, state)
    open_list = []
    tie_breaker = itertools.count()
    
    # Initialization: calculate the initial heuristic using the passed function
    initial_h = heuristic_fn(start_state)
    heapq.heappush(open_list, (initial_h, next(tie_breaker), start_state))
    
    came_from = {start_state: None}
    g_score = {start_state: 0}
    
    nodes_expanded = 0
    
    while open_list:
        _, _, current = heapq.heappop(open_list)
        
        # Check if we have reached the goal
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
        
        for neighbor, move_name in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                
                # Calculate the new f_score by calling the parameterized function
                h_score = heuristic_fn(neighbor)
                f_score = tentative_g_score + h_score
                
                came_from[neighbor] = (current, move_name)
                heapq.heappush(open_list, (f_score, next(tie_breaker), neighbor))
                
    return None, nodes_expanded, -1

if __name__ == "__main__":
    print("--- 8-PUZZLE: GENERATION AND RESOLUTION ---\n")
    
    # 1. Guaranteed initial state generation
    initial_board = generate_solvable_state()
    print("Generated Initial Configuration:")
    print_board(initial_board)
    print(f"Initial Manhattan Heuristic: {manhattan_distance(initial_board)}")
    print("-" * 40)
    
    # 2. Execute A* passing the Manhattan distance as heuristic
    print("Starting A* guided by Manhattan Distance...\n")
    path, expanded, optimal_cost = a_star(initial_board, heuristic_fn=manhattan_distance)
    
    # 3. Debugging (Print the step-by-step path)
    if path is not None:
        print(f"Solution found! Expanded {expanded} nodes.\n")
        print("Move trace:")
        for step, (move, state) in enumerate(path, 1):
            print(f"Step {step}: Move '{move}'")
            print_board(state)
            
        print("=" * 50)
        print(" DATASET ENTRY - READY FOR EXPORT ")
        print("=" * 50)
        print(f"[*] INPUT (Initial Configuration) : {initial_board}")
        print(f"[*] TARGET (Optimal Total Cost)    : {optimal_cost}")
        print("=" * 50)
    else:
        print("Error: No solution found (mathematically illegal).")