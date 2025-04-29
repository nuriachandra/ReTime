import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# data = np.load("./synthetic_data/test.npy")
data = np.load("./synthetic_data/gp.npy")

caseS = [10, 20, 30, 40]
# caseS = [1, 3, 5]
# caseS = [i for i in range(100)]

sns.set(style="whitegrid", font_scale=4)
sns.set_palette("Set2")

xgrid = np.arange(data.shape[-1])
for cdx in caseS:
    fig, axs = plt.subplots(1, 1, figsize=(10, 12), dpi=100)
    axs, idx = [axs], 0
    axs[idx].scatter(xgrid, data[cdx], lw=4)
    axs[idx].plot(xgrid, data[cdx], lw=4)
    plt.tight_layout()
    plt.show()
