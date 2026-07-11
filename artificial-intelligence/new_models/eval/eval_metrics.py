import os
import sys
import torch
import csv

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
from models.delta.delta_model import DeltaPuzzleNet
from models.multitask.multitask_model import MultiTaskDeltaPuzzleNet
from models.embedding.embedding_model import EmbeddingPuzzleNet
from data.dataset import get_dataloaders
from data.delta_dataset import get_delta_dataloaders
from data.embedding_dataset import get_embedding_dataloaders
from environment import GRID_SIZE_8, TARGET_POS_8

def compute_all_models_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running All Models Metrics Analysis on: {device}\n")

    _, _, test_loader_pos = get_dataloaders()
    _, _, test_loader_delta = get_delta_dataloaders()
    _, _, test_loader_emb = get_embedding_dataloaders()

    pos_model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    pos_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth"), map_location=device, weights_only=True))
    pos_model.eval()

    delta_model = DeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    delta_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "delta", "best_delta_model.pth"), map_location=device, weights_only=True))
    delta_model.eval()

    mt_model = MultiTaskDeltaPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    mt_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "multitask", "best_multitask_model.pth"), map_location=device, weights_only=True))
    mt_model.eval()

    emb_model = EmbeddingPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    emb_model.load_state_dict(torch.load(os.path.join(root_dir, "models", "embedding", "best_embedding_model.pth"), map_location=device, weights_only=True))
    emb_model.eval()

    mae_pos, mae_delta, mae_mt, mae_emb = 0.0, 0.0, 0.0, 0.0
    total = len(test_loader_pos)

    with torch.no_grad():
        for (in_p, tgt), (in_d, _), (in_e, _) in zip(test_loader_pos, test_loader_delta, test_loader_emb):
            in_p, tgt = in_p.to(device), tgt.to(device)
            in_d, in_e = in_d.to(device), in_e.to(device)
            true_c = tgt.item()

            p_pred = pos_model(in_p).item()
            d_pred = delta_model(in_d).item()
            mt_pred = mt_model(in_d, return_aux=False).item()
            e_pred = emb_model(in_e).item()

            mae_pos += abs(p_pred - true_c)
            mae_delta += abs(d_pred - true_c)
            mae_mt += abs(mt_pred - true_c)
            mae_emb += abs(e_pred - true_c)

    print("=" * 95)
    print("           COMPLETE OFFLINE EVALUATION SUMMARY (1,000 TEST SAMPLES)")
    print("=" * 95)
    print(f"{'Metric':<30} | {'Positional':<13} | {'Delta':<13} | {'Multi-Task':<13} | {'Embedding':<13}")
    print("-" * 95)
    print(f"{'Mean Absolute Error (MAE)':<30} | {mae_pos/total:<13.4f} | {mae_delta/total:<13.4f} | {mae_mt/total:<13.4f} | {mae_emb/total:<13.4f}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    compute_all_models_metrics()
