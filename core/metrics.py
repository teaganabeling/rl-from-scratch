import pandas as pd

def save_metrics(metrics, filename="metrics.csv"):
    """Save episode metrics to CSV."""
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)

def save_trial_data(config, filename="trial_data.csv"):
    """Save hyperparameters and environment info to CSV."""
    trial_data = {
        "Environment": [config["env"]["name"]],
        "Learning rate": [config["agent"]["learning_rate"]],
        "Gamma": [config["agent"]["gamma"]],
        "Alpha": [config["agent"]["alpha"]],
        "Number of episodes": [config["training"]["n_episodes"]],
        "Max steps": [config["training"]["max_steps"]],
    }

    if config["env"]["name"] == "GridWorld" and "cell_rewards" in config["gridworld"]:
        trial_data.update({
            "Grid size": [config["gridworld"]["grid_size"]],
            "Reward for free space": [config["gridworld"]["cell_rewards"]["free"]],
            "Reward for goal": [config["gridworld"]["cell_rewards"]["goal"]]
        })

    df = pd.DataFrame(trial_data)
    df.to_csv(filename, index=False)