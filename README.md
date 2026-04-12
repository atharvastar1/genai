# FORGE-GRPO: Self-Improving Reward Logic

An implementation of autonomous reward function design using Large Language Models to optimize Reinforcement Learning agents.

## 🎯 Project Overview
This project demonstrates a **Closed-Loop AI System** that automatically designs objective functions for RL agents. Instead of humans manually tuning rewards (e.g., "try 0.1 for velocity, 0.5 for angle"), this system uses a **Teacher-Student** model:
1.  **Teacher (LLM)**: Proposes a physics-based reward function (Python code).
2.  **Student (PPO Agent)**: Learns to balance the pole using that function.
3.  **Evaluator**: Provides physical diagnostics (Stability, Centering, Failure Angle) back to the Teacher to improve the next design iteration.

## 🚀 Presentation Features
*   **Live Physics Visualizer**: Watch the agent replay its best performance in real-time.
*   **Academic Multi-Metrics**: Tracks stability (angular variance) and precision (centering) beyond just the raw score.
*   **Group Comparison**: Visualizes the "Genetic Race" between 3 different AI-generated reward strategies.
*   **Narrative Feedback**: Shows how the AI "reasons" through physical failures.

## 🛠️ Technical Stack
*   **RL Engine**: Stable-Baselines3 (PPO)
*   **Environment**: Gymnasium (CartPole-v1)
*   **Backend**: Flask (Python 3.11)
*   **Frontend**: HTML5 Canvas & Chart.js

## 📝 Academic Connection
This is a simplified implementation of **FORGE** and **GRPO** (Group Relative Policy Optimization). GRPO is a cutting-edge technique used in models like DeepSeek-R1 to allow for self-improving reasoning by comparing groups of candidate outputs.

## 🚦 Quick Start
1.  Run `./setup.sh` to install dependencies.
2.  Run `./run.sh` to launch the dashboard.
3.  Access at `http://localhost:8080`.
