from envs.gridworld_env import GridworldEnv

from core.config import load_config
config = load_config()

import time
import pandas as pd

if config["env"]["name"] == "GridWorld":
    from run_gridworld import run
    run(config)
elif config["env"]["name"] == "CartPole-v1":
    from run_cartpole import run
    run(config)
else:
    raise ValueError(f"Unknown environment: {config['env']['name']}")

