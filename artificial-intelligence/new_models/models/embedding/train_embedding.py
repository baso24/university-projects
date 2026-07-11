import os
import sys
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

from models.embedding.embedding_model import EmbeddingPuzzleNet
from data.embedding_dataset import get_embedding_dataloaders
from environment import GRID_SIZE_8

def train_embedding_model():
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 20
    MIN_DELTA = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding model training will run on: {device}")
    print(f"Number of epochs: {EPOCHS} | Early Stopping Patience: {PATIENCE}")

    train_loader, val_loader, _ = get_embedding_dataloaders(batch_size=64)

    model = EmbeddingPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    current_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_directory, "best_embedding_model.pth")
    plot_path = os.path.join(current_directory, "embedding_loss_curve.png")

    history_train_loss = []
    history_val_loss = []

    print("\nStarting Embedding model training loop...\n")
    epochs_pbar = tqdm(range(EPOCHS), desc="Embedding Training", unit="epoch")

    for epoch in epochs_pbar:
        # training phase
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epochs_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Loss': f'{avg_val_loss:.4f}',
            'Best Val': f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A',
            'LR': f'{current_lr:.6f}',
            'Patience': f'{patience_counter}/{PATIENCE}'
        })

        # Early stopping logic
        if avg_val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[!] Early Stopping triggered at epoch {epoch + 1}: Validation loss did not improve for {PATIENCE} consecutive epochs.")
                break

    print("\nEmbedding model training finished. Generating Loss Curve")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label='Training Loss')
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label='Validation Loss')
    
    plt.title('EmbeddingPuzzleNet: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(plot_path)
    print(f"Loss curve saved to {os.path.basename(plot_path)}")
    plt.close()

if __name__ == "__main__":
    train_embedding_model()
