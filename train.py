import datetime
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.model_utils import create_model
from training.training_utils import do_early_stopping_ckpt, eval_model, plot_result, set_seed
from utils import create_data_loaders, load_data


def train(model, train_loader, val_loader, config, device):
    output_dir = Path(config.output_dir)
    epochs = config.get("epochs", 50)
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 10)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True)
    history = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf"), "patience": patience, "epoch": -1}
    stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        history["epoch"] = epoch

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()

            if config.get("gradient_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({"loss": train_loss / train_batches})

        avg_train_loss = train_loss / train_batches
        history["train_loss"].append(avg_train_loss)

        avg_val_loss, *_ = eval_model(model, criterion, val_loader, device)
        history["val_loss"].append(avg_val_loss)

        if config.wandb.use:
            wandb.log({"val_loss": avg_val_loss})
            wandb.log({"train_loss": avg_train_loss})

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.3e}, Val Loss: {avg_val_loss:.3e}")

        scheduler.step(avg_val_loss)

        stopping_counter = do_early_stopping_ckpt(model, optimizer, history, output_dir, stopping_counter)
        if stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    pickle.dump(history, open(output_dir / "training_history.pkl", "wb"))
    plot_result(history, output_dir)

    return history


def eval_iterative(model, test_loader, device):
    criterion = nn.MSELoss()

    # iterate throught the possible number of recurrences and get the best results
    best_mae = np.inf
    best_r = 1
    for r in range(1, getattr(model, "max_recurrence", 1) + 1):
        avg_test_loss, all_preds, all_targets = eval_model(model, criterion, test_loader, device)
        print(f"Recurrence {r}: Test Loss: {avg_test_loss:.3e}")

        horizon_mse = np.mean((all_preds - all_targets) ** 2, axis=0)
        for h, mse in enumerate(horizon_mse):
            print(f"Recurrence {r}: Horizon {h + 1} MSE: {mse:.3e}")

        mean_horizon_mae = np.mean(np.sum(np.abs(all_preds - all_targets), axis=1))

        print(f"Recurrence {r}: MAE {mean_horizon_mae:.3e}")

        if mean_horizon_mae < best_mae:
            best_mae = mean_horizon_mae
            best_test_loss = avg_test_loss
            best_horizon_mse = horizon_mse
            best_r = r

    return {
        "val_loss": best_test_loss,
        "horizon_mse": best_horizon_mse.tolist(),
        "mean_horizon_mae": float(best_mae),
        "best_r": best_r,
    }


@hydra.main(config_path="configs", config_name="BaseConfig.yml")
def main(cfg: DictConfig):
    output_dir = Path(cfg.get("output_dir", "output"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.model + "_" + os.path.splitext(os.path.basename(cfg.data_path))[0]
    output_dir = output_dir / f"{experiment_name}_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    print("\n" + "*=" * 50)
    print(OmegaConf.to_container(cfg, resolve=True))
    print("*=" * 50 + "\n")

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.proj)
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    if cfg.set_seed:
        print("setting seed")
        set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_data, val_data, test_data = load_data(cfg)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, cfg)

    print("Creating model...")
    model = create_model(cfg=cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")

    print("Starting training...")
    _ = train(model, train_loader, val_loader, cfg, device)

    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating model...")
    eval_results = eval_iterative(model, test_loader, device)
    pickle.dump(eval_results, open(output_dir / "eval_results.pkl", "wb"))

    if cfg.wandb.use:
        wandb.summary["val_loss"] = eval_results["val_loss"]
        wandb.summary["val_best_mae"] = eval_results["mean_horizon_mae"]
        wandb.summary["val_best_r"] = eval_results["best_r"]
        wandb.finish()

    print("Training and evaluation completed!")
    print(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
