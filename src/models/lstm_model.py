"""
LSTM Model for Stock Price Prediction - Anti-Overfit Version

This module contains the LSTM neural network architecture for predicting
stock prices with improved regularization to reduce overfitting.

Author: Tech Challenge Phase 4 - Revised
Date: 2025
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime

# ===========================
# LSTM Model
# ===========================
class LSTMModel(nn.Module):
    """
    LSTM Neural Network for time series prediction with stronger regularization
    """
    def __init__(self, 
                 input_size=4, 
                 hidden_size=30, 
                 num_layers=1, 
                 output_size=1, 
                 dropout=0.35):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout after LSTM
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(0.3)  # additional dropout
        
        # Output layer
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last timestep
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc1_dropout(out)
        out = self.fc2(out)
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            x = x.to(next(self.parameters()).device)
            predictions = self.forward(x)
            return predictions.cpu().numpy()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_path = f"{save_path}.pth"
        torch.save(self.state_dict(), model_path)
        print(f"✓ Model saved to: {model_path}")
        
        model_metadata = {
            'model_info': self.get_model_info(),
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__
        }
        if metadata:
            model_metadata.update(metadata)
        metadata_path = f"{save_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        return model_path, metadata_path
    
    @classmethod
    def load_model(cls, model_path, device='cpu'):
        metadata_path = str(model_path).replace('.pth', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_info = metadata['model_info']
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
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Model weights loaded from: {model_path}")
        print(f"✓ Model moved to device: {device}")
        return model

# ===========================
# Trainer Class
# ===========================
class LSTMTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, criterion, optimizer, 
            num_epochs, scheduler=None, early_stopping_patience=15,
            checkpoint_path='models/checkpoints/checkpoint.pth'):
        
        epochs_without_improvement = 0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if scheduler:
                scheduler.step(val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
        
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.6f}")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


