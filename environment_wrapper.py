import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, reward_func_code):
        super().__init__(env)
        # Create a local scope with necessary utilities
        self.local_scope = {"np": np, "abs": abs, "min": min, "max": max}
        
        # Define the reward function from the code string
        # The expected code is "def reward_fn(obs):\n    ..."
        try:
            exec(reward_func_code, self.local_scope)
            self.reward_fn = self.local_scope.get("reward_fn")
            if not self.reward_fn:
                 raise ValueError("Function 'reward_fn' not found in code.")
        except Exception as e:
            print(f"Error compiling reward function: {e}")
            # Fallback to default reward (1.0 for every step if not failed)
            self.reward_fn = lambda obs: 1.0

    def reward(self, reward):
        # We ignore the original reward and use the custom one
        # CartPole observation: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        obs = self.env.unwrapped.state if hasattr(self.env.unwrapped, 'state') else None
        if obs is None:
             # Fallback if state is not easily accessible
             # Gymnasium returns obs from step() which we can use
             # But RewardWrapper.reward(reward) only gives original reward.
             # We should probably use a standard Wrapper instead of RewardWrapper
             # to have access to the observation.
             return reward
             
        try:
            custom_reward = self.reward_fn(obs)
            return float(custom_reward)
        except Exception as e:
            # print(f"Runtime error in reward function: {e}")
            return 0.0

class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, reward_func_code):
        super().__init__(env)
        self.local_scope = {"np": np, "abs": abs, "min": min, "max": max}
        try:
            exec(reward_func_code, self.local_scope)
            self.reward_fn = self.local_scope.get("reward_fn")
            if not self.reward_fn:
                 raise ValueError("Function 'reward_fn' not found in code.")
        except Exception as e:
            print(f"Error compiling reward function: {e}")
            self.reward_fn = lambda obs: 1.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            # Use our custom reward function
            reward = float(self.reward_fn(obs))
        except Exception as e:
            reward = 0.0
        return obs, reward, terminated, truncated, info
