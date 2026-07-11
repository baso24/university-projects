import os
import sys
import time
import torch

# Add new_models and repository root to sys.path
_current = os.path.abspath(__file__)
while True:
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    if os.path.basename(_parent) == 'new_models':
        sys.path.append(_parent)
        sys.path.append(os.path.dirname(_parent))
        root_dir = _parent
        break
    _current = _parent

from models.positional.model import PuzzleNet
from data.dataset import get_dataloaders
from environment import GRID_SIZE_8

def evaluate_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Offline evaluation will run on: {device}")

    # load only the test loader
    _, _, test_loader = get_dataloaders()

    # instantiate the model and load the saved weights
    model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    model_path = os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model weights loaded successfully from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find model weights at {model_path}.")
        return

    # set the model to evaluation mode
    model.eval()
    
    total_samples = len(test_loader)
    total_absolute_error = 0.0
    overestimated_samples = 0
    error_by_depth = {
        'Easy (0-10 moves)': [],
        'Medium (11-20 moves)': [],
        'Hard (21-31 moves)': []
    }
    
    total_inference_time = 0.0
    
    # print of the first 10 samples just to show some results for example
    print("\n" + "=" * 70)
    print("        OFFLINE EVALUATION: FIRST 10 SAMPLES")
    print("=" * 70)
    print(f"{'Sample':<10} | {'A* Cost (Real)':<18} | {'Network Prediction':<18} | {'Error ':<18}")
    print("-" * 70)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # we start the timer, we do de prediction and we end the timer
            start_time = time.perf_counter()
            prediction = model(inputs)
            end_time = time.perf_counter()
            
            # we add the time of the prediction to the total inference time
            total_inference_time += (end_time - start_time)

            # extract the numerical values from the tensors
            true_cost = targets.item()
            pred_cost = prediction.item()

            # we compute the absolute error for MAE
            absolute_error = abs(true_cost - pred_cost)
            total_absolute_error += absolute_error

            # we count how many times the network overstimate the cost
            if pred_cost > (true_cost + 0.001):
                overestimated_samples += 1

            # we compute the error distribution by difficulty
            # difficulty is based on the number of moves from the optimal solution
            if true_cost <= 10:
                error_by_depth['Easy (0-10 moves)'].append(absolute_error)
            elif true_cost <= 20:
                error_by_depth['Medium (11-20 moves)'].append(absolute_error)
            else:
                error_by_depth['Hard (21-31 moves)'].append(absolute_error)

            # print the first 10 samples just to show some results for example
            if i < 10:
                print(f"#{i+1:<9} | {true_cost:<18.1f} | {pred_cost:<18.4f} | {absolute_error:<10.4f}")
    
    # we compute the mean absolute error
    mae = total_absolute_error / total_samples
    
    # we compute the overestimation percentage
    overestimation_rate = (overestimated_samples / total_samples) * 100
    
    # throughput of the network, how many predictions the network does in one second
    predictions_per_second = total_samples / total_inference_time

    # print the final structured report
    print("\n" + "=" * 70)
    print(f"        FINAL EVALUATION REPORT ON TEST SET ({total_samples} samples)")
    print("=" * 70)
    
    print("\n1. GLOBAL ACCURACY")
    print(f"   Mean Absolute Error (MAE): {mae:.4f} moves")
    
    print("\n2. OVERESTIMATION RATE")
    print(f"   Inadmissible estimates: {overestimated_samples} / {total_samples} ({overestimation_rate:.2f}%)")
        
    print("\n3. ERROR DISTRIBUTION BY DIFFICULTY OF THE PUZZLE")
    for category, errors in error_by_depth.items():
        if len(errors) > 0:
            avg_bucket_error = sum(errors) / len(errors)
            print(f"   {category:<25}: {avg_bucket_error:.4f} moves (based on {len(errors)} samples)")
        else:
            print(f"   {category:<25}: No samples in this range")

    print("\n4. COMPUTATIONAL PERFORMANCE")
    print(f"   Total Inference Time (all test set): {total_inference_time:.4f} seconds")
    print(f"   Speed of the network (predictions/second): {predictions_per_second:.0f}")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    evaluate_offline()