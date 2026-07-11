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

from models.positional.model import PuzzleNet
from models.small.small_model import SmallPuzzleNet
from data.dataset import get_dataloaders
from environment import GRID_SIZE_8, GRID_SIZE_15

def train_distillation():
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    ALPHA = 0.5  # Weighting factor between Hard Loss (true cost) and Soft Loss (teacher prediction)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Knowledge Distillation will run on: {device}")
    print(f"Number of epochs: {EPOCHS} | Alpha (Hard vs Soft Loss weight): {ALPHA}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=256)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 1. Load Teacher Model (PuzzleNet)
    teacher_model = PuzzleNet(grid_size=GRID_SIZE_8).to(device)
    teacher_weights_path = os.path.join(root_dir, "models", "positional", "best_puzzle_model.pth")
    
    try:
        teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device, weights_only=True))
        teacher_model.eval()
        print(f"Teacher model (PuzzleNet) weights loaded successfully from {os.path.basename(teacher_weights_path)}")
    except FileNotFoundError:
        print(f"Error: Could not find teacher weights at {teacher_weights_path}.")
        return

    # 2. Instantiate Student Model (SmallPuzzleNet)
    student_model = SmallPuzzleNet(grid_size=GRID_SIZE_8).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    student_model_path = os.path.join(current_directory, "best_small_puzzle_model.pth")
    plot_path = os.path.join(current_directory, "small_loss_curve.png")

    history_train_loss = []
    history_val_loss = []

    print("\nStarting Knowledge Distillation training loop...\n")
    epochs_pbar = tqdm(range(EPOCHS), desc="Distillation Progress", unit="epoch")

    for epoch in epochs_pbar:
        # Training phase
        student_model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get Teacher predictions (Soft Targets) without gradients
            with torch.no_grad():
                teacher_targets = teacher_model(inputs)
            
            # Student forward pass
            student_outputs = student_model(inputs)
            
            # Combined Loss: Alpha * Hard Loss + (1 - Alpha) * Soft Loss
            hard_loss = criterion(student_outputs, targets)
            soft_loss = criterion(student_outputs, teacher_targets)
            loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # Validation phase
        student_model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                student_outputs = student_model(inputs)
                loss = criterion(student_outputs, targets)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)

        # Update progress bar
        epochs_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Loss': f'{avg_val_loss:.4f}',
            'Best Val': f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student_model.state_dict(), student_model_path)

    print("\nDistillation completed. Generating Loss Curve")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), history_train_loss, label='Distillation Training Loss')
    plt.plot(range(1, EPOCHS + 1), history_val_loss, label='Validation Loss')
    
    plt.title('SmallPuzzleNet (Distilled): Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(plot_path)
    print(f"Loss curve saved to {os.path.basename(plot_path)}")
    plt.close()

if __name__ == "__main__":
    train_distillation()
