from envs.gridworld_env import GridworldEnv
from algorithms.tabular_qn import TabularQN
from algorithms.sarsa import SARSA
from core.metrics import save_metrics, save_trial_data
import time

def run(config):
    # Create environment
    grid_size = config["gridworld"].get("grid_size", "6x6")  # fallback to "6x6" if not specified
    env = GridworldEnv(grid_size)

    # Create agent
    alg = config["gridworld"]["algorithm"]

    if alg == "TabularQN":
        agent = TabularQN(
            nrow=env.nrow,
            ncol=env.ncol,
            n_actions=env.action_space.n
        )
    elif alg == "SARSA":
        agent = SARSA(
            nrow=env.nrow,
            ncol=env.ncol,
            n_actions=env.action_space.n
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['gridworld']['algorithm']}")

    # Training parameters
    n_episodes = config["training"]["n_episodes"]
    max_steps = config["training"]["max_steps"]
    epsilon_start = config["agent"]["epsilon_start"]
    epsilon_end = config["agent"]["epsilon_end"]
    epsilon_decay = config["agent"]["epsilon_decay"]

    # Training loop
    epsilon = epsilon_start

    # Initialize metrics storage
    metrics = {
                'Episode': [],
                'Steps': [],
                'Reward': [],
                'Epsilon': []
            }

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        start_displaying_at_episode = config["training"].get("start_display", 0)  # fallback to 0 if not specified

        while not done and step < max_steps:
            if episode >= start_displaying_at_episode:
                    env.render()  # Visualize current state
                    time.sleep(0.01)  # Slowdown for visualization

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Learn from experience
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

        # Log metrics
        metrics['Episode'].append(episode+1)
        metrics['Steps'].append(step)
        metrics['Reward'].append(total_reward)
        metrics['Epsilon'].append(epsilon)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if start_displaying_at_episode <= episode:
            print(f"Episode {episode+1}/{n_episodes} "
                f"| Steps: {step} "
                f"| Reward: {total_reward:.2f} "
                f"| Epsilon: {epsilon:.3f}")

    env.render()  # Final state
    print("Training complete")

    # Save maps and Q-table
    env.save_reward_map_as_csv("data/reward_map.csv")
    agent.save_q_table_as_csv("data/q_table.csv")
    # Save general trial data
    save_trial_data(config, "data/trial_data.csv")
    # Save results
    save_metrics(metrics, "data/metrics.csv")
