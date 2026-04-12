import gymnasium as gym
from stable_baselines3 import PPO
from environment_wrapper import ObservationWrapper

def train_agent(reward_code, total_timesteps=15000):
    # Setup environment with custom reward
    base_env = gym.make("CartPole-v1", render_mode=None)
    env = ObservationWrapper(base_env, reward_code)
    
    # Simple PPO agent
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate performance
    eval_env = gym.make("CartPole-v1", render_mode=None)
    total_reward = 0
    episodes = 5
    
    # Diagnostic stats
    failure_positions = []
    failure_angles = []
    sample_trajectory = []
    abs_angles = []
    abs_positions = []
    
    for i in range(episodes):
        obs, info = eval_env.reset()
        done = False
        step_count = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, float_reward, terminated, truncated, info = eval_env.step(action)
            total_reward += float_reward
            done = terminated or truncated
            
            # Store physics snapshots
            abs_positions.append(abs(obs[0]))
            abs_angles.append(abs(obs[2]))
            
            # Record trajectory for the first episode only
            if i == 0 and step_count < 200:
                sample_trajectory.append(obs.tolist())
                step_count += 1
                
            if terminated: # Failure state
                failure_positions.append(obs[0])
                failure_angles.append(obs[2])
            
    avg_reward = total_reward / episodes
    
    # Advanced Academic Metrics
    avg_fail_pos = sum(failure_positions)/len(failure_positions) if failure_positions else 0
    avg_fail_angle = sum(failure_angles)/len(failure_angles) if failure_angles else 0
    stability = 1.0 - (sum(abs_angles)/len(abs_angles) if abs_angles else 0) / 0.21 # Scale 0-1
    centering = 1.0 - (sum(abs_positions)/len(abs_positions) if abs_positions else 0) / 2.4 # Scale 0-1
    
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
