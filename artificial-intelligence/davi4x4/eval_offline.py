import os
import torch
from davi_model import PuzzleResNet
from davi_utils import encode_state, scramble_from_goal # Aggiunto scramble_from_goal
from environment import GOAL_STATE_15 # Importiamo il TUO goal state

def test_inference():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Esecuzione inferenza su: {device}\n")

    model = PuzzleResNet(hidden_dim=256, num_blocks=4).to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "davi_model.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Errore: davi_model.pth non trovato.")
        return

    test_boards = [
        (0, GOAL_STATE_15),
        (1, scramble_from_goal(1)),
        (3, scramble_from_goal(3)),
        (5, scramble_from_goal(5)),
        (10, scramble_from_goal(10)),
        (15, scramble_from_goal(15)),
        (20, scramble_from_goal(20)),
        (25, scramble_from_goal(25)),
        (30, scramble_from_goal(30)),
        (35, scramble_from_goal(35)),
        (40, scramble_from_goal(40)),
        (45, scramble_from_goal(45)),
        (50, scramble_from_goal(50)),
        (55, scramble_from_goal(55)),
        (60, scramble_from_goal(60))
    ]

    print("-" * 55)
    print(f"{'Mosse REALI dal Goal':<20} | {'Costo STIMATO dalla Rete':<30}")
    print("-" * 55)

    with torch.no_grad():
        for real_dist, board in test_boards:
            x = encode_state(board).unsqueeze(0).to(device)
            predicted_cost = max(0.0, model(x).item())
            print(f"Scramble: {real_dist:<10} | {predicted_cost:.2f}")

if __name__ == "__main__":
    test_inference()