py -m pip install --upgrade pip
# py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
# py -m pip install torch
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
py -m pip install pytest numpy wandb matplotlib seaborn scipy
py -m pip install tqdm hydra-core pyqt5

py -m pip install "python-lsp-server[all]" --no-cache-dir
py -m pip install ruff
py -m pip uninstall autopep8 -y
py -m pip install python-lsp-ruff --no-cache-dir
py -m pip install pre-commit
