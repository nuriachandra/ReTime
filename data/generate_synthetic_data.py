#!/usr/bin/env python3
"""
Generate sample time series data for testing the BaseTimeTransformer model.
This script creates synthetic time series data with various patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Optional, Union, Tuple

def generate_constant(length: int, value: float = 1.0, noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a constant time series with optional noise.
    
    Args:
        length: Number of time steps to generate
        value: The constant value
        noise_level: Standard deviation of Gaussian noise to add
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    # Create constant array
    series = np.ones(length) * value
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, length)
        series = series + noise
    
    return series

def generate_sine_wave(length: int, amplitude: float = 1.0, frequency: float = 0.1, 
                      phase: float = 0.0, noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a sine wave time series with optional noise.
    
    Args:
        length: Number of time steps to generate
        amplitude: Amplitude of the sine wave
        frequency: Frequency of the sine wave (cycles per time step)
        phase: Phase offset in radians
        noise_level: Standard deviation of Gaussian noise to add
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    t = np.arange(length)
    series = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, length)
        series = series + noise
    
    return series

def generate_linear_trend(length: int, slope: float = 0.01, intercept: float = 0.0, 
                         noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a linear trend time series with optional noise.
    
    Args:
        length: Number of time steps to generate
        slope: Slope of the linear trend
        intercept: Y-intercept of the linear trend
        noise_level: Standard deviation of Gaussian noise to add
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    t = np.arange(length)
    series = slope * t + intercept
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, length)
        series = series + noise
    
    return series

def generate_seasonal(length: int, period: int = 24, amplitude: float = 1.0, 
                    pattern: str = 'sine', noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a seasonal time series with optional noise.
    
    Args:
        length: Number of time steps to generate
        period: Number of time steps in one seasonal cycle
        amplitude: Amplitude of the seasonal pattern
        pattern: Type of seasonal pattern ('sine', 'square', 'triangle', 'sawtooth')
        noise_level: Standard deviation of Gaussian noise to add
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    t = np.arange(length)
    
    if pattern == 'sine':
        # Sine wave pattern
        series = amplitude * np.sin(2 * np.pi * t / period)
    elif pattern == 'square':
        # Square wave pattern
        series = amplitude * (2 * (np.mod(t, period) < period/2) - 1)
    elif pattern == 'triangle':
        # Triangle wave pattern
        series = 2 * amplitude / period * (period/2 - np.abs(np.mod(t, period) - period/2)) - amplitude/2
    elif pattern == 'sawtooth':
        # Sawtooth pattern
        series = 2 * amplitude * (np.mod(t, period) / period) - amplitude
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, length)
        series = series + noise
    
    return series

def generate_random_walk(length: int, drift: float = 0.0, volatility: float = 1.0, 
                        start_value: float = 0.0) -> np.ndarray:
    """
    Generate a random walk time series.
    
    Args:
        length: Number of time steps to generate
        drift: Drift term (mean of the random increments)
        volatility: Volatility (standard deviation of the random increments)
        start_value: Initial value of the random walk
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    # Generate random increments
    increments = np.random.normal(drift, volatility, length)
    
    # Compute cumulative sum to get the random walk
    series = np.cumsum(increments)
    
    # Add the start value
    series = series + start_value
    
    return series

def generate_ar_process(length: int, ar_params: List[float], mean: float = 0.0, 
                       noise_level: float = 1.0, burn_in: int = 100) -> np.ndarray:
    """
    Generate an autoregressive (AR) process.
    
    Args:
        length: Number of time steps to generate
        ar_params: List of AR coefficients
        mean: Mean of the process
        noise_level: Standard deviation of the noise
        burn_in: Number of initial values to discard
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    # Order of the AR process
    p = len(ar_params)
    
    # Initialize longer series to account for burn-in
    total_length = length + burn_in
    series = np.zeros(total_length)
    
    # Initialize with random values
    series[:p] = np.random.normal(mean, noise_level, p)
    
    # Generate the series
    for t in range(p, total_length):
        # Compute AR component
        ar_component = sum(ar_params[i] * series[t-i-1] for i in range(p))
        
        # Add noise
        noise = np.random.normal(0, noise_level)
        
        # Combine components
        series[t] = mean + ar_component + noise
    
    # Discard burn-in period
    series = series[burn_in:]
    
    return series

def generate_complex_pattern(length: int) -> np.ndarray:
    """
    Generate a complex time series with multiple patterns combined.
    
    Args:
        length: Number of time steps to generate
    
    Returns:
        numpy array of shape (length,) containing the time series
    """
    # Generate components
    trend = generate_linear_trend(length, slope=0.01)
    daily_seasonal = generate_seasonal(length, period=24, amplitude=2.0)
    weekly_seasonal = generate_seasonal(length, period=168, amplitude=5.0)  # 24*7=168 hours in a week
    noise = np.random.normal(0, 0.5, length)
    
    # AR component
    ar = generate_ar_process(length, ar_params=[0.8, -0.2], noise_level=0.1)
    
    # Combine components
    series = trend + daily_seasonal + weekly_seasonal + ar + noise
    
    return series

def create_batch_sequences(series: np.ndarray, n_seqs: int, seq_length: int, 
                          horizon_length: int, stride: int = 1) -> np.ndarray:
    """
    Create a batch of sequences from a single time series.
    
    Args:
        series: Input time series
        n_seqs: Number of sequences to create
        seq_length: Length of each input sequence
        horizon_length: Length of each target sequence
        stride: Step size between consecutive sequences
    
    Returns:
        numpy array of shape (n_seqs, seq_length + horizon_length)
    """
    # Calculate total length per sequence
    total_length = seq_length + horizon_length
    
    # Check if we have enough data
    if len(series) < total_length:
        raise ValueError(f"Series length ({len(series)}) is less than required length ({total_length})")
    
    # Generate starting indices with the given stride
    start_indices = [i * stride for i in range(n_seqs)]
    
    # Ensure the last sequence fits within the series
    last_start_idx = start_indices[-1]
    if last_start_idx + total_length > len(series):
        raise ValueError(f"Series is too short for requested n_seqs and stride. " 
                         f"Need length of at least {last_start_idx + total_length}.")
    
    # Create sequences
    sequences = np.array([series[idx:idx+total_length] for idx in start_indices])
    
    return sequences

def plot_series(series: np.ndarray, title: str = 'Time Series', save_path: Optional[str] = None):
    """
    Plot a time series.
    
    Args:
        series: Time series data
        title: Plot title
        save_path: Path to save the plot (if None, the plot is displayed)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate sample time series data')
    parser.add_argument('--pattern', type=str, default='constant', 
                        choices=['constant', 'sine', 'linear', 'seasonal', 
                                'random_walk', 'ar', 'complex'],
                        help='Pattern of the time series')
    parser.add_argument('--n_seqs', type=int, default=32, 
                        help='Number of sequences to generate')
    parser.add_argument('--seq_length', type=int, default=128, 
                        help='Length of each input sequence')
    parser.add_argument('--horizon_length', type=int, default=24, 
                        help='Length of the prediction horizon')
    parser.add_argument('--stride', type=int, default=1, 
                        help='Stride between consecutive sequences')
    parser.add_argument('--noise', type=float, default=0.1, 
                        help='Noise level')
    parser.add_argument('--output', type=str, default='data/time_series.npy', 
                        help='Output file path')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot the generated time series')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Calculate the required time series length based on n_seqs, seq_length, horizon_length, and stride
    min_required_length = ((args.n_seqs - 1) * args.stride) + args.seq_length + args.horizon_length
    
    print(f"Calculating required time series length...")
    print(f"Based on n_seqs={args.n_seqs}, seq_length={args.seq_length}, "
          f"horizon_length={args.horizon_length}, and stride={args.stride}")
    print(f"Minimum required length: {min_required_length}")
    
    # Add some buffer to make visualization nicer
    time_series_length = min_required_length + 50
    
    # Generate time series data based on the selected pattern
    print(f"Generating {args.pattern} time series of length {time_series_length}...")
    
    if args.pattern == 'constant':
        series = generate_constant(time_series_length, value=5.0, noise_level=args.noise)
    elif args.pattern == 'sine':
        series = generate_sine_wave(time_series_length, amplitude=2.0, frequency=0.01, noise_level=args.noise)
    elif args.pattern == 'linear':
        series = generate_linear_trend(time_series_length, slope=0.01, noise_level=args.noise)
    elif args.pattern == 'seasonal':
        series = generate_seasonal(time_series_length, period=48, amplitude=3.0, noise_level=args.noise)
    elif args.pattern == 'random_walk':
        series = generate_random_walk(time_series_length, drift=0.01, volatility=0.1)
    elif args.pattern == 'ar':
        series = generate_ar_process(time_series_length, ar_params=[0.8, -0.2, 0.1], noise_level=args.noise)
    elif args.pattern == 'complex':
        series = generate_complex_pattern(time_series_length)
    else:
        raise ValueError(f"Unknown pattern: {args.pattern}")
    
    # Create batch of sequences
    print(f"Creating {args.n_seqs} sequences with stride {args.stride}...")
    sequences = create_batch_sequences(
        series, 
        args.n_seqs, 
        args.seq_length, 
        args.horizon_length,
        args.stride
    )
    print(f"Generated sequences shape: {sequences.shape}")
    
    # Save the data
    np.save(args.output, sequences.astype(np.float32))
    print(f"Data saved to {args.output}")
    
    # Plot the data if requested
    if args.plot:
        plot_path = os.path.splitext(args.output)[0] + '.png'
        plot_series(series, title=f'Generated {args.pattern.capitalize()} Time Series', save_path=plot_path)
        print(f"Plot saved to {plot_path}")
        
        # Plot a sample sequence
        if len(sequences.shape) > 1:
            sample_idx = 0
            sample_seq = sequences[sample_idx]
            sample_plot_path = os.path.splitext(args.output)[0] + f'_sample_sequence.png'
            plt.figure(figsize=(12, 6))
            plt.plot(sample_seq)
            plt.axvline(x=args.seq_length, color='r', linestyle='--')
            plt.title(f'Sample Sequence (Input: 0-{args.seq_length}, Target: {args.seq_length}-{args.seq_length+args.horizon_length})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.savefig(sample_plot_path)
            plt.close()
            print(f"Sample sequence plot saved to {sample_plot_path}")
    
    # Display some statistics
    print(f"Time series statistics:")
    print(f"  Mean: {np.mean(series):.4f}")
    print(f"  Std Dev: {np.std(series):.4f}")
    print(f"  Min: {np.min(series):.4f}")
    print(f"  Max: {np.max(series):.4f}")
    print(f"Output shape: {sequences.shape} ({args.n_seqs} sequences, each {args.seq_length + args.horizon_length} long)")

if __name__ == "__main__":
    main()