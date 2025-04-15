"""
Training script for BaseTimeTransformer model.
Usage: python train_transformer.py config.yaml
"""

import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from BaseTransformer import BaseTimeTransformer, BaseTimeTransformerConfig
from utils import load_data, create_data_loaders

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Training function
def train(model, train_loader, val_loader, config, device):
    # Set up output directory
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Training parameters
    epochs = config.get('epochs', 50)
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    patience = config.get('patience', 10)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    # Early stopping counter
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # Move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({'loss': train_loss / train_batches})
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for x_batch, y_batch in progress_bar:
                # Move to device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                
                # Update metrics
                val_loss += loss.item()
                val_batches += 1
                progress_bar.set_postfix({'loss': val_loss / val_batches})
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Check if this is the best model
        if avg_val_loss < history['best_val_loss']:
            print(f"Validation loss improved from {history['best_val_loss']:.6f} to {avg_val_loss:.6f}, saving model...")
            history['best_val_loss'] = avg_val_loss
            
            # Save the model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': {k: v for k, v in vars(model.config).items()}
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
        
        # Early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    return history


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BaseTimeTransformer model')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration file
    with open(args.config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Set up output directory
    output_dir = config_dict.get('output_dir', 'output')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config_dict['output_dir'] = output_dir
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)
    
    # Set the seed for reproducibility
    set_seed(config_dict.get('seed', 42))
    
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() and config_dict.get('use_gpu', True) else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    train_data, val_data, test_data = load_data(config_dict)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, config_dict)
    
    # Create model configuration
    model_config = BaseTimeTransformerConfig(
        block_size=config_dict.get('block_size', 1024),
        n_layer=config_dict.get('n_layer', 12),
        n_head=config_dict.get('n_head', 12),
        n_embd=config_dict.get('n_embd', 768),
        h=config_dict.get('h', 2),
        dropout=config_dict.get('dropout', 0.0),
        bias=config_dict.get('bias', False)
    )
    
    # Create model
    print("Creating model...")
    model = BaseTimeTransformer(model_config)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Train the model
    print("Starting training...")
    history = train(model, train_loader, val_loader, config_dict, device)
    
    # Load the best model for evaluation
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    print("Training completed!")
    print(f"All outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
