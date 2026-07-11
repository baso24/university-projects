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

from models.delta.delta_model import DeltaPuzzleNet
from data.delta_dataset import get_delta_dataloaders
from environment import GRID_SIZE_8

def evaluate_delta_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Delta Offline Evaluation running on: {device}")

    _, _, test_loader = get_delta_dataloaders()

    model = DeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    model_path = os.path.join(root_dir, "models", "delta", "best_delta_model.pth")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Delta Model weights loaded successfully from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print("Error: Could not find delta model weights.")
        return

    model.eval()
    
    total_samples = len(test_loader)
    total_absolute_error = 0.0
    overestimated_samples = 0
    error_by_depth = {
        'Easy (0-10 moves)': [],
        'Medium (11-20 moves)': [],
        'Hard (21-31 moves)': []
    }
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            
            true_cost = targets.item()
            pred_cost = prediction.item()
            absolute_error = abs(true_cost - pred_cost)
            total_absolute_error += absolute_error

            if pred_cost > (true_cost + 0.001):
                overestimated_samples += 1

            if true_cost <= 10:
                error_by_depth['Easy (0-10 moves)'].append(absolute_error)
            elif true_cost <= 20:
                error_by_depth['Medium (11-20 moves)'].append(absolute_error)
            else:
                error_by_depth['Hard (21-31 moves)'].append(absolute_error)

    mae = total_absolute_error / total_samples
    overestimation_rate = (overestimated_samples / total_samples) * 100

    print("\n" + "=" * 70)
    print(f"        DELTA MODEL OFFLINE EVALUATION REPORT ({total_samples} samples)")
    print("=" * 70)
    print(f"\n1. DISTANZA MEDIA DI ERRORE (MAE): {mae:.4f} mosse")
    print(f"\n2. TASSO DI SOVRASTIMA: {overestimation_rate:.2f}% ({overestimated_samples}/{total_samples} campioni)")
    print("\n3. ERRORE MEDIO PER DIFFICOLTÀ:")
    for category, errors in error_by_depth.items():
        if len(errors) > 0:
            avg_bucket_error = sum(errors) / len(errors)
            print(f"   {category:<25}: {avg_bucket_error:.4f} mosse")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    evaluate_delta_offline()
