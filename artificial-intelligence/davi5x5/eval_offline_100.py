import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from davi_model import DAVI_Ultra
from davi_utils import encode_states_fast

def test_inference():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running Offline Evaluation on: {device}\n")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "davi_model_5x5.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(current_directory, "davi_model.pth")
    dataset_path = os.path.join(current_directory, "test_dataset_100.json")

    model = DAVI_Ultra().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Unable to find '{model_path}'. Make sure training is completed.")
        return

    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Unable to find dataset '{dataset_path}'. Run the generator first.")
        return

    # Grouping boards by scramble_depth
    problems_by_depth = {}
    for item in dataset:
        d = item["scramble_depth"]
        if d not in problems_by_depth:
            problems_by_depth[d] = []
        problems_by_depth[d].append(item["board"])

    print(f"Dataset loaded: {len(dataset)} total configurations.\n")

    print("=" * 85)
    print(f"{'Depth':<8} | {'Samples':<8} | {'Avg Opt Cost':<14} | {'Avg Pred Cost':<14} | {'MAE':<10} | {'Overestimation Rate (%)':<22}")
    print("-" * 85)

    all_abs_errors = []
    all_overestimations = []
    total_samples = 0

    with torch.no_grad():
        for depth in sorted(problems_by_depth.keys()):
            boards = problems_by_depth[depth]
            num_boards = len(boards)
            
            x_batch = encode_states_fast(boards, device)
            preds = torch.clamp(model(x_batch).squeeze(1), min=0.0).cpu().numpy()
            
            opt_cost = float(depth)
            
            abs_errors = [abs(pred - opt_cost) for pred in preds]
            overestimates = [1 if pred > opt_cost else 0 for pred in preds]
            
            avg_pred = float(preds.mean())
            mae = float(sum(abs_errors) / num_boards)
            overestimation_rate = float(sum(overestimates) / num_boards * 100.0)
            
            all_abs_errors.extend(abs_errors)
            all_overestimations.extend(overestimates)
            total_samples += num_boards
            
            print(f"{depth:<8} | {num_boards:<8} | {opt_cost:<14.1f} | {avg_pred:<14.2f} | {mae:<10.2f} | {overestimation_rate:>21.2f}%")

    print("=" * 85)
    if total_samples > 0:
        global_mae = sum(all_abs_errors) / total_samples
        global_overest_rate = (sum(all_overestimations) / total_samples) * 100.0
        print(f"\n📊 GLOBAL METRICS (All {total_samples} samples combined):")
        print(f"   • Mean Absolute Error (MAE) respect to optimal cost: {global_mae:.2f}")
        print(f"   • Overestimation Rate:                             {global_overest_rate:.2f}%\n")

if __name__ == "__main__":
    test_inference()
