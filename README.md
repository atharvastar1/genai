# 🦾 FORGE-GRPO: Autonomous Reward Engineering Pipeline
### *Mini-Project: Self-Improving RL via Group Relative Policy Optimization*

---

## 🌟 Executive Summary
This project implements a **Closed-Loop Generative AI System** designed to solve complex control tasks (`Acrobot-v1`) without human-led reward shaping. By leveraging a **Teacher-Student** paradigm, the system autonomously iterates through reward function designs, evaluates their physical efficacy, and uses **GRPO** (Group Relative Policy Optimization) logic to refine its reasoning.

## 🧠 Key Innovation: Autonomous Reward Engineering
Classic RL requires a human to manually "tune" reward weights. This project replaces the human with an **LLM Teacher** that:
1.  **Drafts** 3 competing reward strategies.
2.  **Evaluates** them in parallel on separate Student agents.
3.  **Refines** the best logic based on physical diagnostics (Mechanical Energy, Height, Stability).

## 🏗️ Technical Architecture
The system is composed of four integrated layers:

1.  **Generative Layer (LLM Teacher)**: Uses Prompt Engineering and iterative critique to propose Python-based reward functions.
2.  **Simulation Layer (PPO Student)**: Utilizes `Stable-Baselines3` to train agents on the proposed functions.
3.  **Optimization Layer (GRPO Loop)**: Implements parallel execution and advantage-based ranking to identify superior strategies within a group.
4.  **Presentation Layer (Web Dashboard)**: A Flask-based interface for real-time visualization of physics trajectories and academic metrics.

## 🚀 Recent Performance Upgrades (V2.0)
*   **Parallel Execution**: Implemented multiprocessing to train candidate groups 3.0x faster.
*   **Hyperparameter Tuning**: Optimized PPO entropy and GAE lambda for better convergence on swing-up tasks.
*   **Robust Fallbacks**: Integrated physical baseline rewards (Energy Pumping) for environments with constrained API access.

## 🛠️ Installation & Setup
To ensure a smooth evaluation, please follow these steps:

1.  **Environment Preparation**:
    ```bash
    chmod +x setup.sh run.sh
    ./setup.sh
    ```
2.  **Launch the Dashboard**:
    ```bash
    ./run.sh
    ```
3.  **Access**: Open `http://localhost:8080` to view the live optimization progress.

## 📄 Submission Contents
*   `optimizer.py`: The core GRPO group optimization logic.
*   `trainer.py`: High-performance PPO training wrapper.
*   `reward_generator.py`: Agentic reward drafting engine.
*   `history.json`: Comprehensive research log of all 20+ simulation iterations.
*   `project_report.md`: Technical analysis of the system's evolution.

---
**Author**: [User Name]  
**Topic**: Generative AI in Control Theory & Reinforcement Learning  
**Framework**: FORGE-GRPO Implementation
