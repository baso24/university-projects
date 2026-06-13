import heapq
import random
import itertools

# Definiamo lo stato finale per l'8-puzzle (3x3)
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GRID_SIZE = 3

# Pre-calcoliamo le coordinate (x, y) target per ottimizzare la Manhattan distance
TARGET_POS = {val: (i % GRID_SIZE, i // GRID_SIZE) for i, val in enumerate(GOAL_STATE)}

def print_board(state):
    """Stampa la tupla come una griglia 3x3 leggibile."""
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

def get_inversions(state):
    """Calcola il numero di inversioni per il check di parità."""
    state_no_zero = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(state_no_zero)):
        for j in range(i + 1, len(state_no_zero)):
            if state_no_zero[i] > state_no_zero[j]:
                inversions += 1
    return inversions

def generate_solvable_state():
    """Genera una configurazione iniziale casuale garantendo che sia risolvibile."""
    state_list = list(range(9))
    random.shuffle(state_list)
    
    # Per una griglia con larghezza dispari (3x3), il puzzle è risolvibile 
    # se e solo se il numero di inversioni è pari.
    if get_inversions(state_list) % 2 != 0:
        # Se è dispari, scambiamo due tessere non-zero per invertire la parità
        idx1, idx2 = 0, 1
        if state_list[idx1] == 0: idx1 = 2
        if state_list[idx2] == 0: idx2 = 2
        state_list[idx1], state_list[idx2] = state_list[idx2], state_list[idx1]
        
    return tuple(state_list)

def manhattan_distance(state):
    """Calcola la distanza di Manhattan per lo stato corrente."""
    dist = 0
    for i, val in enumerate(state):
        if val != 0:
            curr_x, curr_y = i % GRID_SIZE, i // GRID_SIZE
            targ_x, targ_y = TARGET_POS[val]
            dist += abs(curr_x - targ_x) + abs(curr_y - targ_y)
    return dist

def get_neighbors(state):
    """Genera gli stati figli legali muovendo la casella vuota (0)."""
    neighbors = []
    zero_idx = state.index(0)
    zero_x, zero_y = zero_idx % GRID_SIZE, zero_idx // GRID_SIZE
    
    # Movimenti possibili (dx, dy): Su, Giù, Sinistra, Destra
    moves = {
        'Up': (0, -1),
        'Down': (0, 1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }
    
    for move_name, (dx, dy) in moves.items():
        new_x, new_y = zero_x + dx, zero_y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            new_idx = new_y * GRID_SIZE + new_x
            
            # Creiamo il nuovo stato scambiando lo zero con la tessera target
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move_name))
            
    return neighbors

def a_star(start_state, heuristic_fn):
    """
    Implementazione di A*. 
    Accetta in input lo stato iniziale e una funzione euristica (es. Manhattan o Rete Neurale).
    """
    # Coda di priorità: (f_score, counter, stato)
    open_list = []
    tie_breaker = itertools.count()
    
    # Inizializzazione: calcoliamo l'euristica iniziale tramite la funzione passata come parametro
    initial_h = heuristic_fn(start_state)
    heapq.heappush(open_list, (initial_h, next(tie_breaker), start_state))
    
    came_from = {start_state: None}
    g_score = {start_state: 0}
    
    nodes_expanded = 0
    
    while open_list:
        _, _, current = heapq.heappop(open_list)
        
        # Check se abbiamo raggiunto il goal
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
                
                # Calcolo del nuovo f_score chiamando la funzione parametrizzata
                h_score = heuristic_fn(neighbor)
                f_score = tentative_g_score + h_score
                print(g_score[neighbor], h_score, f_score)
                
                came_from[neighbor] = (current, move_name)
                heapq.heappush(open_list, (f_score, next(tie_breaker), neighbor))
                
    return None, nodes_expanded, -1

if __name__ == "__main__":
    print("--- 8-PUZZLE: GENERAZIONE E RISOLUZIONE ---\n")
    
    # 1. Generazione dello stato iniziale garantito
    initial_board = generate_solvable_state()
    print("Configurazione Iniziale Generata:")
    print_board(initial_board)
    print(f"Euristica Manhattan Iniziale: {manhattan_distance(initial_board)}")
    print("-" * 40)
    
    # 2. Esecuzione di A* passando la distanza di Manhattan come euristica
    print("Avvio A* guidato da Distanza di Manhattan...\n")
    path, expanded, optimal_cost = a_star(initial_board, heuristic_fn=manhattan_distance)
    
    # 3. Debugging (Stampa del percorso step-by-step)
    if path is not None:
        print(f"Soluzione trovata! Espansi {expanded} nodi.\n")
        print("Tracciato delle mosse:")
        for step, (move, state) in enumerate(path, 1):
            print(f"Step {step}: Mossa '{move}'")
            print_board(state)
            
        print("=" * 50)
        print(" DATASET ENTRY - PRONTO PER L'ESPORTAZIONE ")
        print("=" * 50)
        print(f"[*] INPUT (Configurazione Iniziale) : {initial_board}")
        print(f"[*] TARGET (Costo Totale Ottimo)    : {optimal_cost}")
        print("=" * 50)
    else:
        print("Errore: Nessuna soluzione trovata (illegale matematicamente).")