import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _plot_series(series: np.ndarray, title: str = "Time Series", save_path: Optional[str] = None):
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_series(series, sequences, args):
    plot_path = os.path.splitext(args.output)[0] + ".png"
    _plot_series(series, title=f"Generated {args.pattern.capitalize()} Time Series", save_path=plot_path)
    print(f"Plot saved to {plot_path}")

    if len(sequences.shape) > 1:
        sample_idx = 0
        sample_seq = sequences[sample_idx]
        sample_plot_path = os.path.splitext(args.output)[0] + "_sample_sequence.png"
        plt.figure(figsize=(12, 6))
        plt.plot(sample_seq)
        plt.axvline(x=args.seq_length, color="r", linestyle="--")
        plt.title(
            f"Sample Sequence (Input: 0-{args.seq_length}, Target: {args.seq_length}-{args.seq_length + args.horizon_length})"
        )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(sample_plot_path)
        plt.close()
        print(f"Sample sequence plot saved to {sample_plot_path}")
