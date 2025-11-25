from envs.cartpole_env import CartpoleEnv

import numpy as np

from algorithms.dqn import DQN
from algorithms.ppo import PPO
from algorithms.a2c import A2C

from core.metrics import save_metrics, save_trial_data

def run(config):
    env = CartpoleEnv(config)

    # Create agent
    alg = config["cartpole"]["algorithm"]

    if alg == "DQN":
        agent = DQN(env.state_space.shape[0], env.action_space.n)
    elif alg == "PPO":
        agent = PPO(env.state_space.shape[0], env.action_space.n)
    elif alg == "A2C":
        agent = A2C(env.state_space.shape[0], env.action_space.n)
    else:
        raise ValueError(f"Unknown algorithm: {config['cartpole']['algorithm']}")
    
    # Training parameters
    n_episodes = config["training"]["n_episodes"]
    max_steps = config["training"]["max_steps"]

    # Training loop

    # Initialize metrics storage
    metrics = {
                'Episode': [],
                'Steps': [],
                'Reward': [],
                'Epsilon': [],
                'Loss': [],
                'Info': []
            }

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_losses = []  # To store losses for this episode

        while not done and step < max_steps:
            if config["cartpole"]["render_mode"] != "None":
                env.render()  # Visualize current state

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if isinstance(agent, (PPO, A2C)):
                agent.store_reward(reward, done)
            else:
                agent.replay_buffer.store(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            total_reward += reward
            step += 1

        epsilon = f"{agent.epsilon:.3f}" if hasattr(agent, "epsilon") else "N/A"
        # Log metrics
        metrics['Episode'].append(episode+1)
        metrics['Steps'].append(step)
        metrics['Reward'].append(total_reward)
        metrics['Epsilon'].append(epsilon)
        metrics['Loss'].append(np.mean(episode_losses) if episode_losses else None)
        metrics['Info'].append(info)

        loss_val = metrics['Loss'][-1]
        loss_str = f"{loss_val:.3f}" if loss_val is not None else "N/A"

        print(f"Episode {episode+1}/{n_episodes} "
            f"| Steps: {step} "
            f"| Reward: {total_reward:.2f} "
            f"| Epsilon: {epsilon} "
            f"| Loss: {loss_str} "
            f"| Info: {info}")
        
        if isinstance(agent, (PPO, A2C)):
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

        
    print("Training complete")
    env.close()

    # Save general trial data
    save_trial_data(config, "data/trial_data.csv")
    # Save results
    save_metrics(metrics, "data/metrics.csv")