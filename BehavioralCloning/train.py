"""
Training script for Behavioral Cloning
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from state_encoder import StateEncoder
from action_encoder import ActionEncoder
from bc_model import BCModel, TransformerBCModel
from dataset import DiplomacyDataset


POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        states, powers, actions = batch
        states = states.to(device)
        actions = actions.squeeze(1).to(device)
        
        optimizer.zero_grad()
        logits = model(states)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == actions).sum().item()
        total += actions.size(0)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            states, powers, actions = batch
            states = states.to(device)
            actions = actions.squeeze(1).to(device)
            
            logits = model(states)
            loss = criterion(logits, actions)
            total_loss += loss.item()
            
            # Top-1 accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == actions).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = logits.topk(5, dim=1)
            top5_correct += (top5_preds == actions.unsqueeze(1)).any(dim=1).sum().item()
            
            total += actions.size(0)
    
    return total_loss / len(dataloader), correct / total, top5_correct / total


def main():
    # Configuration
    config = {
        'data_path': '../data/standard_no_press.jsonl',
        'max_games': 3000,  # Start small, increase if memory allows
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 20,
        'hidden_size': 512,
        'model_type': 'mlp',  # 'mlp' or 'transformer'
        'save_dir': './checkpoints'
    }
    
    print("="*60)
    print("BEHAVIORAL CLONING TRAINING")
    print("="*60)
    print(f"\nConfig: {json.dumps(config, indent=2)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data
    print("\n" + "-"*40)
    print("Loading data...")
    print("-"*40)
    
    games = []
    with open(config['data_path'], 'r') as f:
        for i, line in enumerate(f):
            if i >= config['max_games']:
                break
            games.append(json.loads(line))
            if (i + 1) % 500 == 0:
                print(f"  Loaded {i + 1} games...")
    
    print(f"Total games: {len(games)}")
    
    # Create encoders
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    
    print("\nBuilding vocabulary...")
    action_encoder.build_vocab(games)
    action_encoder.save_vocab(os.path.join(config['save_dir'], 'vocab.json'))
    
    # Split data
    split_idx = int(0.9 * len(games))
    train_games = games[:split_idx]
    val_games = games[split_idx:]
    print(f"Train: {len(train_games)} games, Val: {len(val_games)} games")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DiplomacyDataset(train_games, state_encoder, action_encoder)
    val_dataset = DiplomacyDataset(val_games, state_encoder, action_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    
    # Create model
    print("\n" + "-"*40)
    print("Creating model...")
    print("-"*40)
    
    if config['model_type'] == 'transformer':
        model = TransformerBCModel(
            state_size=state_encoder.state_size,
            vocab_size=action_encoder.vocab_size
        )
    else:
        model = BCModel(
            state_size=state_encoder.state_size,
            vocab_size=action_encoder.vocab_size,
            hidden_size=config['hidden_size']
        )
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type'].upper()}")
    print(f"Parameters: {num_params:,}")
    print(f"State size: {state_encoder.state_size}")
    print(f"Vocab size: {action_encoder.vocab_size}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "-"*40)
    print("Training...")
    print("-"*40)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top5_acc': []
    }
    best_val_acc = 0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_top5_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Top-5: {val_top5_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"  -> Saved best model (acc: {val_acc:.4f})")
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'config': config
    }, os.path.join(config['save_dir'], 'final_model.pt'))
    
    # Plot training curves
    print("\n" + "-"*40)
    print("Saving results...")
    print("-"*40)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].plot(history['val_top5_acc'], label='Val Top-5', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'), dpi=150)
    print(f"Saved: {config['save_dir']}/training_curves.png")
    
    # Save history
    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
        json.dump(history, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {config['save_dir']}/best_model.pt")


if __name__ == '__main__':
    main()
