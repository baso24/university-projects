# eval_online.py

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import a_star, manhattan_distance
from davi_model import PuzzleResNet
from davi_utils import NeuralHeuristic, scramble_from_goal

def evaluate_online():
    # Rilevamento hardware (incluso MPS per Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Esecuzione Online Evaluation su: {device}")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, "davi_model.pth") # o davi_model_final.pth

    # Inizializzazione della ResNet configurata per il training attuale
    model = PuzzleResNet(hidden_dim=256, num_blocks=4).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Errore: Impossibile trovare '{model_path}'. Assicurati che il training sia concluso.")
        return
    
    model.eval()
    neural_h = NeuralHeuristic(model, device)

    # Generazione del test set con scramble_back di 5 in 5 fino a 60
    scramble_depths = list(range(10, 60, 2))
    problems = [(depth, scramble_from_goal(depth)) for depth in scramble_depths]
    
    print(f"Valutazione su {len(problems)} stati, generati con scramble da 0 a 60...\n")

    # Header della tabella per il print a schermo
    print("=" * 110)
    print(f"{'Depth':<6} | {'BASELINE: A* + MANHATTAN':<45} | {'OURS: A* + NEURAL HEURISTIC':<45}")
    print(f"{'':<6} | {'Nodi Esp.':<12} {'Tempo (s)':<12} {'Path Len':<19} | {'Nodi Esp.':<12} {'Tempo (s)':<12} {'Path Len':<19}")
    print("-" * 110)

    tot_nodes_manhattan = 0
    tot_nodes_neural = 0

    for depth, state in problems:
        # --- 1. A* + Manhattan (Optimal Baseline) ---
        start = time.perf_counter()
        
        # ATTENZIONE: per depth > 45, sul 15-puzzle l'A* con Manhattan rischia
        # di espandere milioni di nodi saturando la RAM. Se si blocca, potresti
        # dover inserire un timeout dentro la tua funzione a_star.
        path_m, expanded_m, cost_m = a_star(state, manhattan_distance)
        
        time_m = time.perf_counter() - start
        len_m = len(path_m)
        tot_nodes_manhattan += expanded_m

        # --- 2. A* + Neural Heuristic ---
        start = time.perf_counter()
        path_n, expanded_n, cost_n = a_star(state, neural_h)
        time_n = time.perf_counter() - start
        len_n = len(path_n)
        tot_nodes_neural += expanded_n
        
        # Controllo di ammissibilità: la rete ha trovato il percorso ottimo?
        # Se la rete sovrastima h(n), l'A* perde la garanzia di ottimalità.
        opt_str_n = f"{len_n}" if len_n == len_m else f"{len_n} (Sub-ottimo!)"

        # Stampa la riga di comparazione
        print(f"{depth:<6} | {expanded_m:<12} {time_m:<12.3f} {len_m:<19} | {expanded_n:<12} {time_n:<12.3f} {opt_str_n:<19}")

    # Statistiche Aggregate
    print("=" * 110)
    if tot_nodes_neural > 0:
        reduction = tot_nodes_manhattan / tot_nodes_neural
        print(f"\n🔥 Riduzione complessiva nodi espansi: {reduction:.2f}x a favore della Rete Neurale.\n")

if __name__ == "__main__":
    evaluate_online()