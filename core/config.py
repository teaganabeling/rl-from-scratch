import yaml
from pathlib import Path

def load_config(filename="config.yaml"):
    """
    Load the YAML config file from the project root.
    """
    root = Path(__file__).parent.parent  # core/ -> rl-from-scratch/
    config_path = root / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
