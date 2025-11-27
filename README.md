# rl-from-scratch
Reinforcement learning library from scratch using raw NumPy to implement core algorithms and test them on classic control tasks and custom environments.

---

## Objectives

### Project Goals
- Implement key RL algorithms (Q-Learning, SARSA, DQN, PPO, and A2C).
- Build custom Gridworlds and environment wrappers for CartPole
- Visualize and analyze training metrics through heatmaps and convergence plots

### Constraints & Requirements
- Written entirely in Numpy, with no use of external ML libraries.
- Modular, easily customizable, with room for further expansion.
- Emphasis on clarity and documentation over optimization.

---

## Summary

**Environment:** GridWorld (12x12)  
**Algorithm:** Q-Learning  
**Average Reward:** ≈ -50 to -30  
**Notes & Observations:**  
Learning curve stayed highly negative early on, but slowly trended upward at the end. Q-table likely not fully converged; longer training or editing parameters could improve results.

---

**Environment:** GridWorld (12x12)  
**Algorithm:** SARSA  
**Average Reward:** ≈ -60 to -40  
**Notes & Observations:**  
Performed slightly worse than Q-Learning (on-policy nature → more conservative updates). Stable but slower to learn optimal policy.

---

**Environment:** CartPole-v1  
**Algorithm:** DQN  
**Average Reward:** ≈ 75-80  
**Notes & Observations:**  
Very fast rise and early plateau at max reward. Quick, strong, stable convergence.

---

**Environment:** CartPole-v1  
**Algorithm:** A2C  
**Average Reward:** ≈ 60-70  
**Notes & Observations:**  
A smooth, consistent increase with few oscillations. Began with a low learning rate, which yielded no convergence. After increasing learning-rate in Trial 2, results drastically improved. Experimenting with other values may yield faster convergence.

---

**Environment:** CartPole-v1  
**Algorithm:** PPO  
**Average Reward:** ≈ 78-80  
**Notes & Observations:**  
Smooth convergence; clipped objective prevents instability. After adjusting learning rate from 0.0003 (Trial 1) to 0.0012 (Trial 2), I saw much faster convergence.

---

## Learning Curves

### Q-Learning
![Q-Learning Learning Curve](images/Q-Learning.jpg)

### SARSA
![SARSA Learning Curve](images/SARSA.jpg)

### DQN
![DQN Learning Curve](images/DQN.jpg)

### A2C
![A2C Learning Curve](images/A2C.jpg)

### PPO
![PPO Learning Curve](images/PPO.jpg)

---

## Outcomes

- Validated correctness of RL implementations; each algorithm behaves as expected.
- Confirmed each agent trains over its compatible environment successfully.
- Observed and visualized algorithm trade-offs:
  - DQN → fastest learning and early convergence.
  - PPO → most smooth learning-rate convergence.
  - A2C → moderate speed with higher variance between episodes.
  - Q-Learning and SARSA → slower tabular learning; consistent with expected behavior.

---

## Future Work

In the future, I hope to expand on the project through:
- Deepening my understanding of reinforcement learning and other machine learning methods.
- Refine and release codebase as an educational toolkit for others’ exploration.
- Potentially integrate continuous-action algorithms (i.e. DDPG).
- Apply trained agents to physical control tasks, like balancing an inverted pendulum or Stewart platform.
