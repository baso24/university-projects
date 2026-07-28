import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

from davi_model import DAVI_Ultra
from davi_utils import GOAL_STATE, get_neighbors_fast, encode_states_fast

def generate_curriculum_data(batch_size, max_depth):
    states = []
    for _ in range(batch_size):
        depth = int(random.triangular(1, max_depth, max_depth))
        state = GOAL_STATE
        for _ in range(depth):
            state = random.choice(get_neighbors_fast(state))
        states.append(state)
    return states

def get_target_values(states, target_net, device):
    all_neighbors = []
    state_slices = []
    
    for state in states:
        if state == GOAL_STATE:
            state_slices.append(0)
            continue
        neighbors = get_neighbors_fast(state)
        all_neighbors.extend(neighbors)
        state_slices.append(len(neighbors))
        
    if not all_neighbors:
        return torch.zeros(len(states), device=device)
        
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            encoded_neighbors = encode_states_fast(all_neighbors, device)
            h_vals = target_net(encoded_neighbors).squeeze(1)
        
    split_preds = torch.split(h_vals, state_slices)
    
    targets = []
    for preds, count in zip(split_preds, state_slices):
        if count == 0:
            targets.append(0.0)
        else:
            targets.append(1.0 + torch.min(preds).item())
            
    return torch.tensor(targets, device=device)

def soft_update(local_model, target_model, tau=0.01):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing DAVI 5x5 Training Pipeline on {device}...")
    
    net = DAVI_Ultra().to(device)
    target_net = DAVI_Ultra().to(device)
    target_net.load_state_dict(net.state_dict())
    
    optimizer = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    BATCH_SIZE = 1024
    ITERATIONS = 50000
    SAVE_EVERY = 5000
    
    current_max_depth = 5
    target_max_depth = 80
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERATIONS, eta_min=1e-5)
    start_time = time.time()
    
    for i in range(1, ITERATIONS + 1):
        if i % 1000 == 0 and current_max_depth < target_max_depth:
            current_max_depth += 5
            
        states = generate_curriculum_data(BATCH_SIZE, current_max_depth)
        targets = get_target_values(states, target_net, device)
        encoded_states = encode_states_fast(states, device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = net(encoded_states).squeeze(1)
            loss = criterion(preds, targets)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        soft_update(net, target_net, tau=0.01)
        
        if i % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter {i:06d}/{ITERATIONS} | Loss: {loss.item():.4f} | Depth: {current_max_depth} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            start_time = time.time()
            
        if i % SAVE_EVERY == 0:
            save_path = f"davi_model_5x5_iter_{i}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"--> Saved Checkpoint: {save_path}")

if __name__ == "__main__":
    train()