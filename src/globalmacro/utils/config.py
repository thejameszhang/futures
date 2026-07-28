import os
from pathlib import Path

import yaml

from .models import Future


def load_config(yaml_file: str | Path = "../../tier1.yaml") -> list[Future]:
    """Load symbols data from a single YAML file and return list of Future objects."""
    yaml_file = os.fspath(yaml_file)
    futures = []

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    for symbol, future_data in data.items():
        # Skip comment lines that start with '#'
        if symbol.startswith('#'):
            continue
        futures.append(Future.from_dict(symbol, future_data))

    return futures
