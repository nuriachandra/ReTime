"""
Training script for BaseTimeTransformer model.
Usage: python train_transformer.py config.yaml
"""

import datetime
import json
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import create_data_loaders, create_model, load_data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# TODO refactor this training function
def train(model, train_loader, val_loader, config, device):
    # Set up output directory
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Training parameters
    epochs = config.get("epochs", 50)
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 10)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True)

    # Training history
    history = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf")}

    # Early stopping counter
    early_stopping_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x_batch, y_batch in progress_bar:
            # Move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.get("gradient_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))

            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({"loss": train_loss / train_batches})

        avg_train_loss = train_loss / train_batches
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Valid]")
            for x_batch, y_batch in progress_bar:
                # Move to device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Forward pass
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

                # Update metrics
                val_loss += loss.item()
                val_batches += 1
                progress_bar.set_postfix({"loss": val_loss / val_batches})

        avg_val_loss = val_loss / val_batches
        history["val_loss"].append(avg_val_loss)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Check if this is the best model
        if avg_val_loss < history["best_val_loss"]:
            print(
                f"Validation loss improved from {history['best_val_loss']:.6f} to {avg_val_loss:.6f}, saving model..."
            )
            history["best_val_loss"] = avg_val_loss

            # Save the model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": {k: v for k, v in vars(model.config).items()},
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))

            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")

        # Early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))

    return history


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    test_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Update metrics
            test_loss += loss.item()
            test_batches += 1

            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_test_loss = test_loss / test_batches
    print(f"Test Loss: {avg_test_loss:.6f}")

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MSE for each horizon step
    horizon_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)
    for h, mse in enumerate(horizon_mse):
        print(f"Horizon {h + 1} MSE: {mse:.6f}")

    return {"test_loss": avg_test_loss, "horizon_mse": horizon_mse.tolist()}


@hydra.main(config_path="configs", config_name="BaseTimeTransformer.yml")
def main(cfg: DictConfig):
    output_dir = cfg.get("output_dir", "output")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir

    # Save configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # Set the seed for reproducibility
    set_seed(cfg.get("seed", 42))

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    train_data, val_data, test_data = load_data(cfg)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, cfg)

    # Create model
    print("Creating model...")
    model = create_model(cfg=cfg)
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")

    # Train the model
    print("Starting training...")
    _ = train(model, train_loader, val_loader, cfg, device)

    # Load the best model for evaluation
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the model
    print("Evaluating model...")
    eval_results = evaluate(model, test_loader, device)

    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f)

    print("Training and evaluation completed!")
    print(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
