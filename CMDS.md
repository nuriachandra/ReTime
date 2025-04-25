```shell
py train.py epochs=1 wandb.use=True
```

```shell
py train.py epochs=1 wandb.use=False model=RecurrentTransformer
```

```shell
py data/generate_synthetic_data.py --pattern sine linear constant ar --n_seqs=100 --seq_length=104 --output synthetic_data/test.npy
```

```shell
py data/process_chronos_data.py --dataset monash_covid_deaths --outpath data/chronos_datasets/test.npy
```

```shell
py train.py epochs=1 wandb.use=False model=BaseTimeTransformer data_path=data/chronos_datasets/test.npy output_dir=experiments/chronos_datasets block_size=5122 h=24
```


