Important Note: The figures for results are given seperately in the repository for evaluation as they are automatically saved in Jupyter notebook while running the code.

 Reinforcement Learning: DQN vs PPO in Chef’s Hat Environment

 Project Overview

This project implements and compares two state-of-the-art Reinforcement Learning algorithms:

* Deep Q-Network (DQN)
* Proximal Policy Optimization (PPO)

Both agents are trained and evaluated using a Chef’s Hat GYM simulator environment.

The objective is to analyze learning stability, convergence behavior, and overall performance differences between value-based and policy-based reinforcement learning approaches in a competitive multi-agent setting.

---

Important Note About Environment Setup

Due to **registry and connectivity issues** encountered with the official GitHub installation of the Chef’s Hat environment, this project uses a Chef’s Hat GYM simulator implementation instead.

The simulator replicates:

* Observation structure (405-dimensional state vector)
* Discrete action space
* Sparse rank-based reward mechanism
* Multi-agent competitive dynamics

This ensured stable execution, reproducibility, and consistent training without dependency conflicts or external registry failures.

---

 Objectives

* Implement DQN (value-based RL)
* Implement PPO (policy-gradient RL)
* Train both agents under identical simulator conditions
* Compare:

  * Convergence speed
  * Reward progression
  * Stability
  * Final performance
* Evaluate algorithm suitability for sparse-reward competitive environments

---

Environment Details

* Environment Type: Chef’s Hat GYM Simulator
* Observation Space: 405-dimensional vector
* Action Space: Discrete
* Reward Type: Sparse (based on finishing rank)
* Game Type: Multi-agent competitive card game

---

Algorithms Implemented

Deep Q-Network (DQN)

* Value-based reinforcement learning
* Experience replay buffer
* Target network stabilization
* ε-greedy exploration
* Bellman equation updates

Strengths

* Sample efficient
* Straightforward architecture
* Stable with replay mechanism

Limitations

* Sensitive to sparse rewards
* Q-value overestimation risk
* Less robust in multi-agent competitive dynamics

---

 Proximal Policy Optimization (PPO)

* Policy-gradient method
* Actor-Critic architecture
* Clipped surrogate objective
* Advantage estimation
* On-policy learning

Strengths

* More stable policy updates
* Better exploration
* Handles sparse rewards more effectively
* Stronger convergence behavior

Limitations

* Higher computational cost
* On-policy (less sample efficient)

---

Results Summary

| Metric             | DQN         | PPO    |
| ------------------ | ----------- | ------ |
| Learning Stability | Moderate    | High   |
| Convergence Speed  | Slower      | Faster |
| Reward Consistency | Fluctuating | Smooth |
| Final Performance  | Good        | Strong |

PPO demonstrated more stable learning and superior performance under sparse reward conditions.

---

Installation

```bash
git clone https://github.com/yourusername/rl-chefs-hat-comparison.git
cd rl-chefs-hat-comparison
pip install -r requirements.txt
```

Requirements

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib
* Gymnasium / OpenAI Gym

---

Running the Project

Train DQN

```bash
python train_dqn.py
```

 Train PPO

```bash
python train_ppo.py
```

 Or Run Notebook

```bash
jupyter notebook Reinforcement\ Learning\ Coursework.ipynb
```

---

Project Structure

```
.
├── Reinforcement Learning Coursework.ipynb
├── train_dqn.py
├── train_ppo.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

 Key Insights

* PPO performs better in sparse-reward competitive environments.
* DQN shows higher reward variance due to exploration instability.
* Multi-agent dynamics significantly increase learning complexity.
* Environment stability is critical for reproducible RL experiments.

---

 Coursework Information

Module: Generative AI and Reinforcement Learning
Project Type: Robustness And Generalisation 
Focus: Empirical comparison of DQN and PPO

---

 Author

Ajinkya Deokate
MSc Data Science & Computational Intelligence
Coventry University

---



