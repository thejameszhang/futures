# Overview

This repository contains the code used to create a Global Macroeconomic Dataset, which analagous to the code created in the paper
“Is There a Replication Crisis in Finance?” (Jensen, Kelly, and Pedersen, Journal of Finance 2023) but for global macroassets.

Follow this [link](https://www.overleaf.com/read/jrdvqrqmcwrt#d28fa4) for detailed documentation on the dataset.

The code consists of two self-contained components:

- `globalmacro` is a folder with code that creates datasets of global futures of commodities, bonds, currencies, and equity indices, and more. **Note that the data can be downloaded without running the code.** 

- `analysis` is a folder that contains analysis of the Global Macroeconomic Dataset.

## Get Started

### Setting up a virtual environment after cloning

If you have already cloned this repository, you do not need to run `uv init`. Instead, create and populate the virtual environment directly from the existing `pyproject.toml` / `uv.lock` files:

```bash
uv sync
```

`uv sync` will create a `.venv/` folder (if one does not exist) and install every dependency listed for the project. You can then activate the environment with `source .venv/bin/activate`, or run commands through uv without manual activation using `uv run <command>`.
