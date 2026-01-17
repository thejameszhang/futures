import os
from typing import List

import yaml

from .models import Future

def load_config(yaml_file="../../tier1.yaml") -> List[Future]:
    """Load tier1 symbols data from YAML file and return list of Future objects.
    If yaml_file contains 'tier2', loads from both tier1.yaml and tier2.yaml."""
    yaml_file = os.fspath(yaml_file)
    futures = []
    
    # Determine which files to load
    if 'tier2' in yaml_file:
        # Load both tier1.yaml and tier2.yaml
        yaml_dir = os.path.dirname(yaml_file) if os.path.dirname(yaml_file) else '..'
        if yaml_dir == '.':
            yaml_dir = '..'
        files_to_load = [
            os.path.join(yaml_dir, 'tier1.yaml'),
            os.path.join(yaml_dir, 'tier2.yaml')
        ]
    else:
        # Load only the specified file
        files_to_load = [yaml_file]
    
    # Process each file
    for file_path in files_to_load:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for symbol, future_data in data.items():
            # Skip comment lines that start with '#'
            if symbol.startswith('#'):
                continue
            futures.append(Future.from_dict(symbol, future_data))
    
    return futures
