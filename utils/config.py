import yaml
from typing import Any, Dict
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    """
    A simple wrapper to access dictionary keys as attributes.
    """
    def __init__(self, entries: Dict[str, Any]):
        for key, value in entries.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
