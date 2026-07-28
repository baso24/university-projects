import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from davi_utils import scramble_from_goal

def generate_benchmark():
    # Dataset parameters
    depths = [20, 30, 40, 50, 60]
    samples_per_depth = 200
    
    benchmark_dataset = []
    seen_states = set()
    
    print("Starting Test Set generation \n")
    
    for depth in depths:
        print(f"Generating {samples_per_depth} boards at depth {depth}...")
        count = 0
        
        while count < samples_per_depth:
            board = scramble_from_goal(depth)
            board_tuple = tuple(board)
            
            # Ensure the board is unique across the dataset
            if board_tuple not in seen_states:
                seen_states.add(board_tuple)
                
                # Save the pair (nominal_depth, board_state)
                benchmark_dataset.append({
                    "scramble_depth": depth,
                    "board": board
                })
                count += 1
                
    # Saving to disk in JSON format
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset_1000.json")
    
    with open(output_path, 'w') as f:
        json.dump(benchmark_dataset, f, indent=4)
        
    print("\n Dataset generated successfully!")
    print(f"Saved to: {output_path}") 
    print(f"Total unique states guaranteed: {len(seen_states)}")

if __name__ == "__main__":
    generate_benchmark()