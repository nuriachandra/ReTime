```shell
py train.py epochs=1 wandb.use=True
```

```shell
py train.py epochs=1 wandb.use=False model=RecurrentTransformer injection_type=add
```

```shell
py train.py epochs=1 wandb.use=False model=BaseTimeTransformer data_path=data/chronos_datasets/test.npy output_dir=experiments/chronos_datasets block_size=188 h=24
```

Testing without and with padding
```shell
# without padding
py train.py epochs=1 wandb.use=False block_size=124 data_path=data/synthetic_data/test.npy padding=False out_style=ext

# with padding
py train.py epochs=1 wandb.use=False block_size=624 data_path=data/synthetic_data/test.npy padding=True out_style=ext
```

To test on Alina's mac [will delete after package issues fixed]
```shell
PYTHONPATH=/Users/alina/WilsonLab/ReTime pytest tests/models_tests/test_models.py
```

# Data creation commands
```shell
py data/generate_synthetic_data.py --pattern sine linear constant ar --n_seqs=100 --seq_length=104 --output synthetic_data/test.npy
```

```shell
py data/process_chronos_data.py --dataset monash_covid_deaths --outpath data/chronos_datasets/test.npy --data_column target
```

```shell
py kernel_synth.py --num-series 100 --save_path 'gp_synthetic_data/100_train'
py data/kernel_synth.py --num-series 100000 --save_path 'data/gp_synthetic_data/100k_5-5-25'

py train.py epochs=1 wandb.use=False block_size=624 data_path=data/gp_synthetic_data/100_train.npz padding=True out_style=ext
```


