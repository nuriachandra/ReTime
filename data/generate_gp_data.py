from pathlib import Path

import numpy as np


def generate_rbf_data(bsz, n_points):
    x = np.random.uniform(low=0, high=100, size=n_points)
    ls = np.random.uniform(low=0.05, high=100, size=(bsz, 1, 1))
    sigma = np.random.uniform(low=0.1, high=1, size=(bsz, 1, 1))
    ker = rbf_kernel(x, x, params=(ls, sigma))
    return ker


def rbf_kernel(x1, x2, params):
    ls, sigma = params
    N1, N2 = x1.shape[-1], x2.shape[-1]
    ker = x1[..., :, None] - x2[..., None, :]
    ker = np.exp(-((ker[None] / ls) ** 2.0)) + sigma * np.eye(N1, N2)[None]
    return ker


if __name__ == "__main__":
    bsz, n_points = 50, 100
    output = Path("./synthetic_data/")
    output.mkdir(parents=True, exist_ok=True)
    ker = generate_rbf_data(bsz, n_points)
    L = np.linalg.cholesky(ker)
    z = np.random.uniform(low=0, high=100, size=n_points)
    y = L @ z
    np.save(output / "gp.npy", y.astype(np.float32))
    print(f"Data saved to {output}")
