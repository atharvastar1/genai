import gymnasium as gym
import math
import numpy as np
from stable_baselines3 import PPO

def train_agent(reward_code, total_timesteps=5000):
    # Setup environment: Acrobot-v1 is much more complex than CartPole
    # Obs: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    env = gym.make("Acrobot-v1")
    
    # Custom Reward Wrapper
    class CustomRewardWrapper(gym.Wrapper):
        def __init__(self, env, code_str):
            super().__init__(env)
            # Create a safe namespace for the function
            namespace = {"abs": abs, "sin": math.sin, "cos": math.cos, "np": np}
            exec(code_str, namespace)
            self.reward_fn = namespace["reward_fn"]

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            try:
                # Custom Reward Logic
                custom_reward = float(self.reward_fn(obs))
            except:
                custom_reward = -0.1 # Penalty for broken code
            return obs, custom_reward, terminated, truncated, info

    wrapped_env = CustomRewardWrapper(env, reward_code)
    
    # Train using PPO
    model = PPO("MlpPolicy", wrapped_env, verbose=0, device="cpu")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate performance (standard metric)
    eval_env = gym.make("Acrobot-v1")
    total_reward = 0
    episodes = 5
    
    # Diagnostic collections
    height_achieved = []
    angular_velocities = []
    samples = []

    for i in range(episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, float_reward, terminated, truncated, info = eval_env.step(action)
            total_reward += float_reward
            
            # Acrobot height: -cos(theta1) - cos(theta1+theta2)
            # Higher is better
            height = -obs[0] - (obs[0]*obs[2] - obs[1]*obs[3])
            height_achieved.append(height)
            angular_velocities.append(abs(obs[4]) + abs(obs[5]))
            
            if i == 0: samples.append(obs.tolist())
            done = terminated or truncated

    avg_reward = total_reward / episodes
    
    diagnostics = {
        "avg_reward": avg_reward,
        "failure_summary": f"Pole failed at {avg_fail_angle:.3f} rad. Centering was {centering*100:.1f}%.",
        "sample_trajectory": sample_trajectory,
        "stability_score": f"{stability*100:.1f}%",
        "centering_score": f"{centering*100:.1f}%"
    }
    
    return diagnostics, model

if __name__ == "__main__":
    # Test with default code
    test_code = "def reward_fn(obs): return 1.0"
    score, _ = train_agent(test_code)
    print(f"Test Score: {score}")
