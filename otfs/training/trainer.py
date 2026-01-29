"""
Training Utilities for OTFS Models
===================================

Generic training and validation loops with checkpointing.

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, epochs=100, 
                val_loader=None, scheduler=None, save_path=None, verbose=True):
    """
    Generic training loop
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epochs: Number of epochs
        val_loader: Optional validation loader
        scheduler: Optional learning rate scheduler
        save_path: Path to save best model
        verbose: Print progress
        
    Returns:
        history: Dictionary with training history
    """
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[-1]
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            val_loss = validate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            if scheduler:
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                save_checkpoint(model, save_path, epoch, val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}")
    
    return history


def validate_model(model, val_loader, criterion, device):
    """
    Validation loop
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[-1]
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def save_checkpoint(model, path, epoch, loss):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        path: Save path
        epoch: Current epoch
        loss: Current loss
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, path, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        device: Device
        
    Returns:
        epoch: Epoch number
        loss: Loss value
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)
