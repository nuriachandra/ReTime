import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def do_early_stopping_ckpt(model, optimizer, history, output_dir):
    epoch, avg_val_loss, patience = history["epoch"], history["val_loss"][-1], history["patience"]
    if avg_val_loss < history["best_val_loss"]:
        msg = f"Validation loss improved from {history['best_val_loss']:.6f} to "
        msg += f"to {avg_val_loss:.6f}, saving model..."
        print(msg)
        history["best_val_loss"] = avg_val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "config": {k: v for k, v in vars(model.config).items()},
        }
        torch.save(checkpoint, output_dir / "best_model.pth")

        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
    return early_stopping_counter


@torch.no_grad
def eval_model(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(val_loader, desc="[Valid]")
    for x_batch, y_batch in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        val_loss += loss.item()
        val_batches += 1
        all_targets.append(y_batch.cpu().numpy())
        all_preds.append(y_pred.cpu().numpy())

        progress_bar.set_postfix({"loss": val_loss / val_batches})

    avg_val_loss = val_loss / val_batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_val_loss, all_preds, all_targets


def plot_result(history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig(output_dir / "loss_plot.png")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
