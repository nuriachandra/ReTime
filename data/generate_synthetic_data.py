import os
from typing import List

import numpy as np

from data.data_parser import get_data_parser


def generate_constant(length: int, value: float, **_) -> np.ndarray:
    series = np.ones(length) * value
    return series


def generate_sine_wave(length: int, amplitude: float, frequency: float, phase: float, **_) -> np.ndarray:
    t = np.arange(length)
    series = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return series


def generate_linear_trend(length: int, slope: float, intercept: float, **_) -> np.ndarray:
    t = np.arange(length)
    series = slope * t + intercept
    return series


def generate_seasonal(length: int, period: int, amplitude: float, pattern: str, **_) -> np.ndarray:
    t = np.arange(length)

    if pattern == "sine":
        series = amplitude * np.sin(2 * np.pi * t / period)
    elif pattern == "square":
        series = amplitude * (2 * (np.mod(t, period) < period / 2) - 1)
    elif pattern == "triangle":
        series = 2 * amplitude / period * (period / 2 - np.abs(np.mod(t, period) - period / 2)) - amplitude / 2
    elif pattern == "sawtooth":
        series = 2 * amplitude * (np.mod(t, period) / period) - amplitude
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return series


def generate_random_walk(length: int, drift: float, volatility: float, start_value: float, **_) -> np.ndarray:
    increments = np.random.normal(drift, volatility, length)
    series = np.cumsum(increments)
    series = series + start_value
    return series


def generate_ar_process(
    length: int,
    ar_params: List[float],
    mean: float,
    noise_level: float,
    burn_in,
    **_,
) -> np.ndarray:
    p = len(ar_params)

    total_length = length + burn_in
    series = np.zeros(total_length)

    series[:p] = np.random.normal(mean, noise_level, p)

    for t in range(p, total_length):
        ar_component = sum(ar_params[i] * series[t - i - 1] for i in range(p))
        noise = np.random.normal(0, noise_level)
        series[t] = mean + ar_component + noise

    series = series[burn_in:]

    return series


def generate_complex_pattern(length: int, **_) -> np.ndarray:
    trend = generate_linear_trend(length, slope=0.01, intercept=1.0)
    daily_seasonal = generate_seasonal(length, period=24, amplitude=2.0, pattern="sine")
    weekly_seasonal = generate_seasonal(length, period=168, amplitude=5.0, pattern="sine")  # 24*7=168 hours in a week
    noise = np.random.normal(0, 0.5, length)

    ar = generate_ar_process(length, ar_params=[0.8, -0.2], noise_level=0.1, burn_in=0, mean=0.0)

    series = trend + daily_seasonal + weekly_seasonal + ar + noise

    return series


TSFNs = {
    "constant": generate_constant,
    "sine": generate_sine_wave,
    "linear": generate_linear_trend,
    "seasonal": generate_seasonal,
    "rw": generate_random_walk,
    "ar": generate_ar_process,
    "complex": generate_complex_pattern,
}


def create_batch_sequences(
    n_seqs: int,
    pattern: str,
    args,
) -> np.ndarray:
    min_length = ((n_seqs - 1) * args.stride) + args.seq_length + args.horizon_length
    series = TSFNs[pattern](min_length, **args.__dict__)
    noise = args.noise * np.random.normal(size=series.shape[0])
    series = series + noise
    print("Time series statistics:")
    print(f"  Mean: {np.mean(series):.4f}")
    print(f"  Std Dev: {np.std(series):.4f}")
    print(f"  Min: {np.min(series):.4f}")
    print(f"  Max: {np.max(series):.4f}")

    total_length = args.seq_length + args.horizon_length
    start_indices = [i * args.stride for i in range(n_seqs)]
    last_start_idx = start_indices[-1]
    if last_start_idx + total_length > len(series):
        raise ValueError(
            f"Series is too short for requested n_seqs and stride. "
            f"Need length of at least {last_start_idx + total_length}."
        )

    sequences = np.array([series[idx : idx + total_length] for idx in start_indices])
    return sequences


def main(args):
    print("Calculating required time series length...")
    print(f"Based on n_seqs={args.n_seqs}, seq_length={args.seq_length}, ")
    print(f"horizon_length={args.horizon_length}, and stride={args.stride}")
    print(f"Creating {args.n_seqs} sequences with stride {args.stride}...")
    length = args.seq_length + args.horizon_length
    chunk_sz = args.n_seqs // len(args.pattern)
    sequences = np.zeros(shape=(args.n_seqs, length))
    for idx, pattern in enumerate(args.pattern):
        aux = create_batch_sequences(chunk_sz, pattern, args)
        sequences[idx * chunk_sz : (idx + 1) * chunk_sz, :] = aux
    np.random.shuffle(sequences)
    print(f"Generated sequences shape: {sequences.shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, sequences.astype(np.float32))
    print(f"Data saved to {args.output}")

    text = f"Output shape: {sequences.shape} ({args.n_seqs} sequences, "
    text += f"each {args.seq_length + args.horizon_length} long)"
    print(text)


if __name__ == "__main__":
    parser = get_data_parser()
    args = parser.parse_args()
    main(args)
