import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
#np.random.seed(42)  # Per riproducibilità
LEARNING_RATE = 0.5 # Tasso di apprendimento alto per convergenza rapida su problema semplice
EPOCHS = 10000      # Numero totale di epoche
SNAPSHOTS = [0, 500, 2000, 10000] # Step in cui visualizzare i grafici (Punto 2)

# --- DATI XOR ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Input
y = np.array([[0], [1], [1], [0]])             # Target

# --- FUNZIONI DI ATTIVAZIONE ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x) # Nota: qui assumiamo x sia già l'output della sigmoide

# --- CLASSE RETE NEURALE ---
class XORNetwork:
    def __init__(self):
        # Architettura 2-2-1
        # Pesi Input -> Hidden (2 neuroni, 2 input ciascuno)
        self.W1 = np.random.uniform(size=(2, 2)) 
        self.b1 = np.random.uniform(size=(1, 2))
        
        # Pesi Hidden -> Output (1 neurone, 2 input)
        self.W2 = np.random.uniform(size=(2, 1))
        self.b2 = np.random.uniform(size=(1, 1))
        
        # Per salvare la storia della loss
        self.loss_history = []

    def forward(self, X):
        # Layer Nascosto
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1) # Attivazioni hidden (coordinate nello hidden space)
        
        # Layer Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y):
        # Calcolo dell'errore (MSE derivative chain rule semplificata)
        # Errore rispetto all'output
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        
        # Errore retropropagato all'hidden layer
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)
        
        # Aggiornamento pesi (Gradient Descent)
        self.W2 += self.a1.T.dot(output_delta) * LEARNING_RATE
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * LEARNING_RATE
        self.W1 += X.T.dot(hidden_delta) * LEARNING_RATE
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * LEARNING_RATE

    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y)
        loss = 0.5 * np.mean((y - output) ** 2)
        self.loss_history.append(loss)

# --- VISUALIZZAZIONE ---
def plot_results(net, epoch, ax_surface, ax_hidden):
    # Creazione Grid per la superficie di decisione
    xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass sulla griglia
    z1_grid = np.dot(grid, net.W1) + net.b1
    a1_grid = sigmoid(z1_grid)
    z2_grid = np.dot(a1_grid, net.W2) + net.b2
    preds = sigmoid(z2_grid).reshape(xx.shape)

    # --- PLOT INPUT SPACE ---
    ax_surface.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    ax_surface.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="RdBu", edgecolors='black', s=100, linewidth=2)
    ax_surface.set_title(f"Input Space (Epoch {epoch})")
    ax_surface.set_xlim(-0.6, 1.6)
    ax_surface.set_ylim(-0.6, 1.6)
    
    # Disegno le linee dei neuroni nascosti
    # Equazione neurone: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2
    x_line = np.linspace(-5, 5, 100)
    colors = ['green', 'orange']
    for i in range(2): # Per ogni neurone nascosto
        w_x, w_y = net.W1[0, i], net.W1[1, i]
        b = net.b1[0, i]
        if abs(w_y) > 0.001: # Evita divisione per zero
            y_line = -(w_x * x_line + b) / w_y
            ax_surface.plot(x_line, y_line, color=colors[i], linestyle='--', label=f'Hidden N{i+1}')
    ax_surface.legend(loc='lower right', fontsize='x-small')

    # --- PLOT HIDDEN SPACE ---
    # Calcoliamo dove finiscono i 4 punti XOR nello spazio nascosto
    _, hidden_points = list(zip(*[ (net.forward(x.reshape(1,2)), net.a1[0]) for x in X ]))
    hidden_points = np.array(hidden_points)
    
    ax_hidden.scatter(hidden_points[:,0], hidden_points[:,1], c=y.ravel(), cmap="RdBu", edgecolors='black', s=100, linewidth=2)
    ax_hidden.set_title(f"Hidden Space (Epoch {epoch})")
    ax_hidden.set_xlabel("Activation H1")
    ax_hidden.set_ylabel("Activation H2")
    ax_hidden.set_xlim(-0.6, 1.6)
    ax_hidden.set_ylim(-0.6, 1.6)

    # Disegno la linea di separazione del neurone di output
    # Equazione output: v1*h1 + v2*h2 + c = 0 (dove h1, h2 sono gli assi qui)
    v1, v2 = net.W2[0, 0], net.W2[1, 0]
    c = net.b2[0, 0]
    
    h_line = np.linspace(-5, 5, 100)
    if abs(v2) > 0.001:
        v_line = -(v1 * h_line + c) / v2
        ax_hidden.plot(h_line, v_line, color='black', linestyle='--', linewidth=2, label='Output Sep.')
        
    ax_hidden.legend(loc='lower right', fontsize='x-small')

if __name__ == "__main__":
    
    net = XORNetwork()
    plt.figure(figsize=(10, 8))

    rows = len(SNAPSHOTS)
    cols = 2

    plot_idx = 1
    snapshot_counter = 0

    for epoch in range(EPOCHS + 1):
        # training step
        net.train(X, y)
        
        # Visualizzazione grafica agli step richiesti
        if epoch in SNAPSHOTS:
            # Input Space Plot
            ax1 = plt.subplot(rows, cols, plot_idx)
            # Hidden Space Plot
            ax2 = plt.subplot(rows, cols, plot_idx + 1)
            
            plot_results(net, epoch, ax1, ax2)
            plot_idx += 2

    plt.tight_layout()
    plt.show()

    # --- PLOT LOSS ---
    plt.figure(figsize=(8, 5))
    plt.plot(net.loss_history)
    plt.title("Risk Function Evolution (Loss vs Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()