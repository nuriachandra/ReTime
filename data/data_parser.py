import argparse


def get_data_parser():
    parser = argparse.ArgumentParser(description="Generate sample time series data")
    parser.add_argument("--n_seqs", type=int, default=32, help="Number of sequences to generate")
    parser.add_argument("--seq_length", type=int, default=128, help="Length of each input sequence")
    parser.add_argument("--horizon_length", type=int, default=24, help="Length of the prediction horizon")
    parser.add_argument("--stride", type=int, default=1, help="Stride between consecutive sequences")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    parser.add_argument("--output", type=str, default="data/time_series.npy", help="Output file path")
    parser.add_argument("--plot", action="store_true", help="Plot the generated time series")

    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--frequency", type=float, default=0.01)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--slope", type=float, default=0.01)
    parser.add_argument("--intercept", type=float, default=0.0)
    parser.add_argument("--period", type=float, default=48)
    parser.add_argument(
        "--pattern",
        nargs="+",
        type=str,
        default=["sine"],
        choices=["constant", "sine", "linear", "seasonal", "random_walk", "ar", "complex", "mix"],
        help="Pattern of the time series",
    )
    parser.add_argument("--drift", type=float, default=0.01)
    parser.add_argument("--volatility", type=float, default=1.0)
    parser.add_argument("--start_value", type=float, default=0.0)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--burn_in", type=float, default=100)
    parser.add_argument("--ar_params", nargs="+", type=float, default=[0.8, -0.2, 0.1])
    return parser
