```shell
py train.py epochs=1 wandb.use=True
```

```shell
py data/generate_synthetic_data.py --pattern sine linear constant ar --n_seqs=100 --seq_length=104 --output synthetic_data/test.npy
```
