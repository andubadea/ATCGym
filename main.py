import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import bluesky_gym

bluesky_gym.register_envs()

TRAIN = True
EVAL_EPISODES = 10

if __name__ == "__main__":
    # Create the environment
    env = gym.make('ConflictArt-v0', n_envs = 4, render_mode=None)
    env = VecFrameStack(env, n_stack=4)
    
    # Can check the environment here
    # check_env(env)
    
    obs, info = env.reset()

    # Create the model
    model = A2C("CnnPolicy", env, verbose=1)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(12e4))
        model.save("models/ConflictArt-v0_ppo/model")
        del model
    
    env.close()
    
    # Test the trained model
    model = PPO.load("models/ConflictArt-v0_ppo/model", env=env)
    env = gym.make('ConflictArt-v0', render_mode='human')

    for i in range(EVAL_EPISODES):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action[()])

    env.close()