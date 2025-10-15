"""
LSTM Model for Stock Price Prediction

This module contains the LSTM neural network architecture for predicting
stock prices based on historical time series data.

Author: Tech Challenge Phase 4
Date: 2025
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for time series prediction
    
    Architecture:
    - LSTM Layer 1 (with return_sequences)
    - Dropout
    - LSTM Layer 2
    - Dropout
    - Fully Connected Layer
    - Output Layer
    """
    
    def __init__(self, 
                 input_size=5, 
                 hidden_size=50, 
                 num_layers=2, 
                 output_size=1, 
                 dropout=0.2):
        """
        Initialize LSTM model
        
        Parameters:
        -----------
        input_size : int
            Number of input features (default: 1 for Close price only)
        hidden_size : int
            Number of features in hidden state (neurons per LSTM layer)
        num_layers : int
            Number of stacked LSTM layers
        output_size : int
            Number of output features (default: 1 for next day prediction)
        dropout : float
            Dropout probability between LSTM layers (0.0 to 1.0)
        """
        super(LSTMModel, self).__init__()
        
        # Store hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0  # Dropout only if multiple layers
        )
        
        # Dropout layer (applied after LSTM output)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        
        # Output layer
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        # Fully connected layer with ReLU
        out = self.fc1(out)
        out = self.relu(out)
        
        # Output layer
        out = self.fc2(out)
        
        return out
    
    def predict(self, x):
        """
        Make predictions (wrapper for forward with eval mode)
        
        Parameters:
        -----------
        x : torch.Tensor or np.ndarray
            Input data of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        np.ndarray
            Predictions as numpy array
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Convert to tensor if numpy array
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            # Ensure correct device
            x = x.to(next(self.parameters()).device)
            
            # Get predictions
            predictions = self.forward(x)
            
            # Convert back to numpy
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def count_parameters(self):
        """
        Count total number of trainable parameters
        
        Returns:
        --------
        int
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """
        Get model architecture information
        
        Returns:
        --------
        dict
            Dictionary with model information
        """
        return {
            'architecture': 'LSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, save_path, metadata=None):
        """
        Save model state dict and metadata
        
        Parameters:
        -----------
        save_path : str
            Path to save the model (without extension)
        metadata : dict, optional
            Additional metadata to save (training metrics, date, etc.)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state dict
        model_path = f"{save_path}.pth"
        torch.save(self.state_dict(), model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Prepare metadata
        model_metadata = {
            'model_info': self.get_model_info(),
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__
        }
        
        # Add custom metadata if provided
        if metadata is not None:
            model_metadata.update(metadata)
        
        # Save metadata as JSON
        metadata_path = f"{save_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path
    
    @classmethod
    def load_model(cls, model_path, device='cpu'):
        """
        Load model from saved state dict
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model (.pth file)
        device : str
            Device to load model to ('cpu' or 'cuda')
            
        Returns:
        --------
        LSTMModel
            Loaded model instance
        """
        # Load metadata to get architecture parameters
        metadata_path = str(model_path).replace('.pth', '_metadata.json')
        # metadata_path = model_path.replace('.pth', '_metadata.json')


        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_info = metadata['model_info']
            
            # Create model with saved architecture
            model = cls(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size'],
                num_layers=model_info['num_layers'],
                output_size=model_info['output_size'],
                dropout=model_info['dropout']
            )
            
            print(f"✓ Model architecture loaded from metadata")
        else:
            print("⚠️ Metadata not found. Using default architecture.")
            model = cls()
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✓ Model weights loaded from: {model_path}")
        print(f"✓ Model moved to device: {device}")
        
        return model


class LSTMTrainer:
    """
    Trainer class for LSTM model with utilities for training and evaluation
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model : LSTMModel
            LSTM model instance
        device : str
            Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Train for one epoch
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        criterion : nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
            
        Returns:
        --------
        float
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """
        Validate model
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function
            
        Returns:
        --------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(self, train_loader, val_loader, criterion, optimizer, 
            num_epochs, scheduler=None, early_stopping_patience=15,
            checkpoint_path='models/checkpoints/checkpoint.pth'):
        """
        Train model for multiple epochs with validation and early stopping
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        num_epochs : int
            Number of epochs to train
        scheduler : torch.optim.lr_scheduler, optional
            Learning rate scheduler
        early_stopping_patience : int
            Number of epochs to wait before early stopping
        checkpoint_path : str
            Path to save best model checkpoint
            
        Returns:
        --------
        dict
            Training history with losses
        """
        print(f"\n{'='*70}")
        print(f"TRAINING LSTM MODEL")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*70}\n")
        
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
            
            # Check if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
# if __name__ == "__main__":
#     # Example of how to use the model
    
#     # Initialize model
#     model = LSTMModel(
#         input_size=5,
#         hidden_size=50,
#         num_layers=2,
#         output_size=1,
#         dropout=0.2
#     )
    
#     # Print model info
#     print("Model Architecture:")
#     print(model)
#     print(f"\nTotal parameters: {model.count_parameters():,}")
    
#     # Test forward pass
#     batch_size = 32
#     seq_length = 60
#     input_size = 5
    
#     dummy_input = torch.randn(batch_size, seq_length, input_size)
#     output = model(dummy_input)
    
#     print(f"\nInput shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}")
    
#     # Save model
#     model.save_model(
#         'models/saved_models/lstm_model',
#         metadata={
#             'lookback_window': 60,
#             'features': ['Close'],
#             'target': 'Close'
#         }
#     )
    
#     # Load model
#     loaded_model = LSTMModel.load_model(
#         'models/saved_models/lstm_model.pth',
#         device='cpu'
#     )
    
#     print("\n✅ Model loaded successfully!")