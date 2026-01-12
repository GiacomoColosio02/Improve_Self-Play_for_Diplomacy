"""
Dataset for Behavioral Cloning
Loads and preprocesses Diplomacy games for training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from state_encoder import StateEncoder
from action_encoder import ActionEncoder

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']


class DiplomacyDataset(Dataset):
    """
    PyTorch Dataset for Diplomacy behavioral cloning.
    
    Each sample is (state, power, action) where:
        - state: encoded board state
        - power: which power is making the move
        - action: the order being issued
    """
    
    def __init__(self, games: List[Dict], state_encoder: StateEncoder, 
                 action_encoder: ActionEncoder, power_filter: Optional[str] = None):
        """
        Args:
            games: List of game dicts
            state_encoder: StateEncoder instance
            action_encoder: ActionEncoder instance (with vocab built)
            power_filter: If set, only include orders from this power
        """
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.power_filter = power_filter
        
        # Extract all (state, power, order) tuples
        self.samples = []
        self._process_games(games)
        
        print(f"Created dataset with {len(self.samples)} samples")
        
    def _process_games(self, games: List[Dict]):
        """Process games into samples."""
        for game in games:
            phases = game.get('phases', [])
            
            for phase in phases:
                phase_name = phase.get('name', '')
                state = phase.get('state', {})
                orders = phase.get('orders', {})
                
                # Skip non-movement phases for now
                if not phase_name.endswith('M'):
                    continue
                
                # Encode state once per phase
                encoded_state = self.state_encoder.encode(state, phase_name)
                
                # Process orders for each power
                for power in POWERS:
                    if self.power_filter and power != self.power_filter:
                        continue
                        
                    power_orders = orders.get(power, [])
                    if power_orders is None:
                        continue
                    
                    power_idx = POWERS.index(power)
                    
                    for order in power_orders:
                        action_idx = self.action_encoder.encode_order(order)
                        
                        # Skip unknown orders
                        if action_idx <= 1:  # PAD or UNK
                            continue
                        
                        self.samples.append({
                            'state': encoded_state,
                            'power': power_idx,
                            'action': action_idx
                        })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['state']),
            torch.LongTensor([sample['power']]),
            torch.LongTensor([sample['action']])
        )


class DiplomacyDataModule:
    """
    Data module that handles loading and splitting data.
    """
    
    def __init__(self, data_path: str, max_games: int = 5000, 
                 batch_size: int = 64, num_workers: int = 0):
        self.data_path = data_path
        self.max_games = max_games
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self):
        """Load data and create datasets."""
        print(f"Loading games from {self.data_path}...")
        
        # Load games
        games = []
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_games:
                    break
                games.append(json.loads(line))
                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1} games...")
        
        print(f"Total games loaded: {len(games)}")
        
        # Build action vocabulary
        print("Building vocabulary...")
        self.action_encoder.build_vocab(games)
        
        # Split into train/val (90/10)
        split_idx = int(0.9 * len(games))
        train_games = games[:split_idx]
        val_games = games[split_idx:]
        
        print(f"Train games: {len(train_games)}, Val games: {len(val_games)}")
        
        # Create datasets
        print("Creating train dataset...")
        self.train_dataset = DiplomacyDataset(
            train_games, self.state_encoder, self.action_encoder
        )
        
        print("Creating val dataset...")
        self.val_dataset = DiplomacyDataset(
            val_games, self.state_encoder, self.action_encoder
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def test_dataset():
    """Test the dataset."""
    print("Testing dataset...")
    
    # Load a few games
    games = []
    with open('../data/standard_no_press.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            games.append(json.loads(line))
    
    print(f"Loaded {len(games)} games")
    
    # Create encoders
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    action_encoder.build_vocab(games)
    
    # Create dataset
    dataset = DiplomacyDataset(games, state_encoder, action_encoder)
    
    # Test a sample
    state, power, action = dataset[0]
    print(f"\nSample 0:")
    print(f"  State shape: {state.shape}")
    print(f"  Power: {POWERS[power.item()]}")
    print(f"  Action: {action_encoder.decode_order(action.item())}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  States: {batch[0].shape}")
    print(f"  Powers: {batch[1].shape}")
    print(f"  Actions: {batch[2].shape}")


if __name__ == '__main__':
    test_dataset()

