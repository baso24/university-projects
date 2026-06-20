import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import PuzzleNet
from dataset import get_dataloaders
from environment import GRID_SIZE_8, GRID_SIZE_15

def train_model():
    LEARNING_RATE = 0.0001
    EPOCHS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}")
    print(f"Number of epochs: {EPOCHS}")

    train_loader, val_loader, _ = get_dataloaders()

    # here we specify the size of the puzzle
    grid_size=GRID_SIZE_8
    model = PuzzleNet(grid_size=grid_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    current_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_directory, "best_model.pth")
    plot_path = os.path.join(current_directory, "loss_curve.png")

    history_train_loss = []
    history_val_loss = []

    print("Starting training loop...\n")

    for epoch in range(EPOCHS):
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

        # print loss for each epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            # save the bestmodel weights
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved better model weights")

    print("\nTraining completed. Generating Loss Curve")

    # plot of train and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), history_train_loss, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), history_val_loss, label='Validation Loss')
    
    plt.title(f'PuzzleNet: {grid_size}x{grid_size} puzzle - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # save the plot
    plt.savefig(plot_path)
    print(f"Loss curve saved to {os.path.basename(plot_path)}")
    plt.show()

if __name__ == "__main__":
    train_model()