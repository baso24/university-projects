import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Funzione di attivazione sigmoid e sua derivata
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

class XORNetwork:
    def __init__(self):
        # Inizializzazione random dei pesi e dei bias
        self.W1 = np.random.uniform(size=(2, 2)) 
        self.b1 = np.random.uniform(size=(1, 2))
        self.W2 = np.random.uniform(size=(2, 1))
        self.b2 = np.random.uniform(size=(1, 1))
        # Per salvare i valori della loss function
        self.loss_history = []

    def forward(self, X):
        # Calcolo esplicito di tutti i layer della rete
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1) 
        self.z2 = self.a1 @ self.W2 + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y, learning_rate):
        # Il simbolo @ indica la moltiplicazione di matrici in numpy, è equivalente a np.dot()
        # Errore rispetto all'output
        output_error = self.output - y
        output_delta = output_error * sigmoid_derivative(self.output)
        
        # Errore retropropagato all'hidden layer
        hidden_error = output_delta @ self.W2.T
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)
        
        # Aggiornamento pesi e bias
        self.W2 -= (self.a1.T @ output_delta) * learning_rate
        self.b2 -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.W1 -= (X.T @ hidden_delta) * learning_rate
        self.b1 -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate):
        output = self.forward(X)
        self.backward(X, y, learning_rate)
        loss = 0.5 * np.mean((y - output) ** 2)
        self.loss_history.append(loss)

# Visualizzazione dei risultati della rete
def plot_results(net, epoch, ax_surface, ax_hidden, ax_act1, ax_act2, X, y):
    # Creazione griglia di punti per visualizzare lo sfondo colorato (decision boundary)
    # np.linspace genera 100 numeri equispaziati tra -5 e 5.
    # np.meshgrid crea due matrici 2D (100x100): 'xx' per le coordinate x e 'yy' per le y.
    xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    # 'xx.ravel()' e 'yy.ravel()' appiattiscono le matrici in vettori 1D (da 100x100 a 10000).
    # np.c_ li concatena colonna per colonna creando una matrice (10000, 2).
    # 'grid' è quindi l'elenco di tutti i punti (x,y) del piano che passeremo alla rete.
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # -- Input layer --
    # Eseguiamo il forward pass su tutta la griglia (10000 punti) per colorare lo sfondo.
    # Usiamo net.forward che calcola tutto e ci restituisce l'output finale.
    preds = net.forward(grid).reshape(xx.shape)
    # Salviamo le attivazioni dell'hidden layer (net.a1) per disegnare le activation maps
    a1_grid = net.a1

    # Plot dello spazio di input con decision boundary
    ax_surface.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    ax_surface.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="RdBu", edgecolors='black', s=100, linewidth=2)
    ax_surface.set_title(f"Input Space (Epoch {epoch})", fontsize=10)
    ax_surface.set_xlabel("Input x1", fontsize=9)
    ax_surface.set_ylabel("Input x2", fontsize=9)
    ax_surface.set_xlim(-0.6, 1.6)
    ax_surface.set_ylim(-0.6, 1.6)
    
    # Disegno le linee dei neuroni dell'hidden layer
    # Equazione neurone: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2
    x_line = np.linspace(-5, 5, 100)
    colors = ['green', 'orange']
    # Per ogni neurone nell'hidden layer
    for i in range(2): 
        w_x, w_y = net.W1[0, i], net.W1[1, i]
        b = net.b1[0, i]
        # Evita divisione per zero
        if abs(w_y) > 0.001: 
            # Calcolo e plotto y per la linea di decisione
            y_line = -(w_x * x_line + b) / w_y
            ax_surface.plot(x_line, y_line, color=colors[i], linestyle='--', label=f'Hidden N{i+1}')
    ax_surface.legend(loc='lower right', fontsize='x-small')

    # -- Hidden layer --
    # Calcoliamo dove finiscono i 4 punti XOR nello spazio nascosto (output del primo layer)
    # Possiamo passare l'intera matrice X (4x2) al forward pass in un colpo solo.
    # La rete calcolerà le attivazioni per tutti e 4 i punti contemporaneamente.
    net.forward(X)
    hidden_points = net.a1
    
    # Plot dell'hidden layer space
    ax_hidden.scatter(hidden_points[:,0], hidden_points[:,1], c=y.ravel(), cmap="RdBu", edgecolors='black', s=100, linewidth=2)
    ax_hidden.set_title(f"Hidden Space (Epoch {epoch})", fontsize=10)
    ax_hidden.set_xlabel("Activation H1", fontsize=9)
    ax_hidden.set_ylabel("Activation H2", fontsize=9)
    ax_hidden.set_xlim(-0.6, 1.6)
    ax_hidden.set_ylim(-0.6, 1.6)

    # Disegno la linea di separazione del neurone di output
    # Equazione output: v1*h1 + v2*h2 + c = 0 -> h2 = -(v1*h1 + c) / v2 (dove h1, h2 sono gli assi qui)
    v1, v2 = net.W2[0, 0], net.W2[1, 0]
    c = net.b2[0, 0]
    h_line = np.linspace(-5, 5, 100)
     # Evita divisione per zero
    if abs(v2) > 0.001:
        # Calcolo e plotto y per la linea di decisione
        v_line = -(v1 * h_line + c) / v2
        ax_hidden.plot(h_line, v_line, color='black', linestyle='--', linewidth=2, label='Output Sep.')
    ax_hidden.legend(loc='lower right', fontsize='x-small')

    # -- Activation maps dei neuroni dell'hidden layer --
    # Activation Neuron 1
    act1 = a1_grid[:, 0].reshape(xx.shape)
    ax_act1.contourf(xx, yy, act1, levels=50, cmap="Greys", vmin=0, vmax=1)
    ax_act1.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="RdBu", edgecolors='white', s=80, linewidth=1.5)
    ax_act1.set_title(f"Activation H1 (Epoch {epoch})", fontsize=10)
    ax_act1.set_xlabel("Input x1", fontsize=9)
    ax_act1.set_ylabel("Input x2", fontsize=9)
    ax_act1.set_xlim(-0.6, 1.6)
    ax_act1.set_ylim(-0.6, 1.6)

    # Activation Neuron 2
    act2 = a1_grid[:, 1].reshape(xx.shape)
    ax_act2.contourf(xx, yy, act2, levels=50, cmap="Greys", vmin=0, vmax=1)
    ax_act2.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="RdBu", edgecolors='white', s=80, linewidth=1.5)
    ax_act2.set_title(f"Activation H2 (Epoch {epoch})", fontsize=10)
    ax_act2.set_xlabel("Input x1", fontsize=9)
    ax_act2.set_ylabel("Input x2", fontsize=9)
    ax_act2.set_xlim(-0.6, 1.6)
    ax_act2.set_ylim(-0.6, 1.6)

# Funzione di aggiornamento dei grafici allo spostamento dello slider
def update(val, slider, net_history, net, ax1, ax2, ax3, ax4, ax_loss, fig, X, y):
    # Prendo valore indice dello slider (numero epoch)
    idx = int(val)
    state = net_history[idx]
    slider.valtext.set_text(f"{state['epoch']}")
    
    # Ripristina pesi per la visualizzazione
    net.W1 = state['W1']
    net.b1 = state['b1']
    net.W2 = state['W2']
    net.b2 = state['b2']
    
    # Pulisci e ridisegna con il nuovo stato della rete
    # Quindi con i pesi e i bias dell'epoch selezionata
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear(); ax_loss.clear()
    plot_results(net, state['epoch'], ax1, ax2, ax3, ax4, X, y)
    
    # Plot Loss
    full_loss = net_history[-1]['loss']
    ax_loss.plot(full_loss, label='Training Loss')
    # Marker corrente
    curr_loss_val = full_loss[state['epoch']]
    ax_loss.scatter(state['epoch'], curr_loss_val, color='red', s=50, zorder=5, label='Current Epoch')
    ax_loss.set_title("Risk Function Evolution", fontsize=10)
    ax_loss.set_xlabel("Epochs", fontsize=9)
    ax_loss.set_ylabel("MSE Loss", fontsize=9)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True)
    
    fig.canvas.draw_idle()

if __name__ == "__main__":    
    
    # Parametri rete e training
    # Se si vuole aumentare il numero di epochs vanno ridefiniti anche gli snapshots...
    LEARNING_RATE = 0.5 
    EPOCHS = 2000      
    SNAPSHOTS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000] 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    y = np.array([[0], [1], [1], [0]])
    net_history = []        
    
    net = XORNetwork()

    # Training della rete e salvataggio degli snapshots
    for epoch in range(EPOCHS + 1):
        net.train(X, y, LEARNING_RATE)
        if epoch in SNAPSHOTS:
            net_history.append({
                'epoch': epoch,
                'W1': net.W1.copy(), 'b1': net.b1.copy(),
                'W2': net.W2.copy(), 'b2': net.b2.copy(),
                'loss': list(net.loss_history)
            })

    # Creazione figura e layout
    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.077, right=0.95, bottom=0.137, top=0.877, hspace=0.336, wspace=0.423)
    fig.suptitle("XOR Network Training Dashboard", fontsize=14, fontweight='bold')
    fig.text(0.5, 0.94, "Legend: Red Points = Class 0 (Target 0), Blue Points = Class 1 (Target 1) | Activation Maps: White=0 (Inactive), Black=1 (Active)", 
             ha='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Creazione griglia per tutti i grafici
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax_loss = fig.add_subplot(gs[1, :])

    # Slider
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Epoch', 0, len(net_history)-1, valinit=0, valstep=1)
    # Funzione di callback per lo slider che richiama update (con il val = valore corrente nello slider)
    slider.on_changed(lambda val: update(val, slider, net_history, net, ax1, ax2, ax3, ax4, ax_loss, fig, X, y))

    # Plot iniziale con val = 0
    update(0, slider, net_history, net, ax1, ax2, ax3, ax4, ax_loss, fig, X, y)
    plt.show()