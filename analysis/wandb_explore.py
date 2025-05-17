import pandas as pd

fpath = "./analysis/results.csv"
df = pd.read_csv(fpath)
target = "val_loss (Min)"
cols = ["model", "n_embd", target]
cols.remove(target)
df = df[df["learning_rate"] > 0.001]
df = df[df["model"] == "RecurrentTransformer"]
dff = df.groupby(cols).min(target).reset_index()
print(dff)
