"""
Behavioral Cloning Model for Diplomacy
Neural network that learns to predict human moves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class BCModel(nn.Module):
    """
    Simple MLP model for behavioral cloning.
    
    Input: encoded game state
    Output: probability distribution over orders
    """
    
    def __init__(self, state_size: int, vocab_size: int, hidden_size: int = 512):
        super().__init__()
        
        self.state_size = state_size
        self.vocab_size = vocab_size
        
        # MLP layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
            
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc4(x)
        return logits
    
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities.
        
        Args:
            x: Input tensor
            temperature: Sampling temperature (1.0 = normal, <1 = more greedy)
            
        Returns:
            Probabilities of shape (batch_size, vocab_size)
        """
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)
    
    def sample_action(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Returns:
            (action_idx, probability)
        """
        probs = self.predict(x, temperature)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, probs.gather(1, action.unsqueeze(1)).squeeze(1)


class TransformerBCModel(nn.Module):
    """
    Transformer-based model for behavioral cloning.
    Better at capturing complex patterns in the state.
    """
    
    def __init__(self, state_size: int, vocab_size: int, 
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Project state to d_model dimensions
        self.input_proj = nn.Linear(state_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, vocab_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
            
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Project to d_model
        x = self.input_proj(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        
        # Output
        logits = self.output_head(x)
        return logits
    
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)


def test_models():
    """Test the models."""
    state_size = 1216
    vocab_size = 5000
    batch_size = 32
    
    # Test MLP model
    print("Testing MLP model...")
    mlp = BCModel(state_size, vocab_size)
    x = torch.randn(batch_size, state_size)
    logits = mlp(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # Test Transformer model
    print("\nTesting Transformer model...")
    transformer = TransformerBCModel(state_size, vocab_size)
    logits = transformer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")


if __name__ == '__main__':
    test_models()
