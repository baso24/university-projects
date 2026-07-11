import os
import sys
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

from environment import GRID_SIZE_8

from models.positional.model import PuzzleNet
from models.small.small_model import SmallPuzzleNet
from models.delta.delta_model import DeltaPuzzleNet
from models.multitask.multitask_model import MultiTaskDeltaPuzzleNet
from models.embedding.embedding_model import EmbeddingPuzzleNet

from data.dataset import get_dataloaders
from data.delta_dataset import get_delta_dataloaders
from data.embedding_dataset import get_embedding_dataloaders

def evaluate_all_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Offline evaluation of all models will run on: {device}\n")

    # Load all test loaders
    print("Loading datasets...")
    _, _, pos_test_loader = get_dataloaders()
    _, _, delta_test_loader = get_delta_dataloaders()
    _, _, emb_test_loader = get_embedding_dataloaders()
    print("Datasets loaded successfully.\n")

    models_config = [
        {
            "name": "Classic Positional",
            "model": PuzzleNet(grid_size=GRID_SIZE_8).to(device),
            "weights": os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth"),
            "loader": pos_test_loader,
        },
        {
            "name": "Small Positional",
            "model": SmallPuzzleNet(grid_size=GRID_SIZE_8).to(device),
            "weights": os.path.join(root_dir, "models", "small", "best_small_puzzle_model.pth"),
            "loader": pos_test_loader,
        },
        {
            "name": "Delta",
            "model": DeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device),
            "weights": os.path.join(root_dir, "models", "delta", "best_delta_model.pth"),
            "loader": delta_test_loader,
        },
        {
            "name": "Multi-Task Delta",
            "model": MultiTaskDeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device),
            "weights": os.path.join(root_dir, "models", "multitask", "best_multitask_model.pth"),
            "loader": delta_test_loader,
        },
        {
            "name": "Embedding",
            "model": EmbeddingPuzzleNet(grid_size=GRID_SIZE_8).to(device),
            "weights": os.path.join(root_dir, "models", "embedding", "best_embedding_model.pth"),
            "loader": emb_test_loader,
        }
    ]

    results = []

    print("=" * 80)
    print("        OFFLINE EVALUATION OVER ALL MODELS")
    print("=" * 80)

    for config in models_config:
        model_name = config["name"]
        model = config["model"]
        weights_path = config["weights"]
        loader = config["loader"]

        print(f"\nEvaluating {model_name}...")
        
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        except FileNotFoundError:
            print(f"  [!] Weights not found at {os.path.basename(weights_path)}. Skipping.")
            continue
        
        model.eval()
        
        total_samples = len(loader)
        total_absolute_error = 0.0
        overestimated_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                prediction = model(inputs)
                
                true_cost = targets.item()
                pred_cost = prediction.item()
                
                total_absolute_error += abs(true_cost - pred_cost)
                
                if pred_cost > (true_cost + 0.001):
                    overestimated_samples += 1
        
        mae = total_absolute_error / total_samples
        overestimation_rate = (overestimated_samples / total_samples) * 100
        
        results.append({
            "name": model_name,
            "mae": mae,
            "overestimation_rate": overestimation_rate,
            "overestimated": overestimated_samples,
            "total": total_samples
        })
        
        print(f"  MAE: {mae:.4f} moves")
        print(f"  Overestimation Rate: {overestimation_rate:.2f}%")

    print("\n\n" + "=" * 80)
    print("        FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model Name':<25} | {'MAE':<15} | {'Overestimation Rate':<25}")
    print("-" * 80)
    for res in results:
        print(f"{res['name']:<25} | {res['mae']:<15.4f} | {res['overestimation_rate']:>6.2f}% ({res['overestimated']}/{res['total']})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    evaluate_all_offline()
