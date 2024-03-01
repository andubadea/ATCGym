import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
import bluesky_gym

bluesky_gym.register_envs()
# check_env(gym.make('ConflictArt-v0', render_mode=None))

RL_MODEL = 'PPO'
IMAGE_MODE = 'rel_rgb'
N_INTRUDERS = 4
IMAGE_SIZE = 128
SEED = 42
NUM_CPU = 4
EVAL_EPISODES = 10
TRAIN = True
TEST = False

class RLTrainer:
    def __init__(self, model:str = 'DQN', image_mode:str = 'rgb', n_intruders:int = 4, image_size:int = 128, 
                 seed:int = 42, num_cpu:int = 4, eval_episodes:int = 10, train:bool = True, test:bool = False):
        self.model=model
        self.image_mode=image_mode
        self.n_intruders=n_intruders
        self.image_size=image_size
        self.seed=42
        self.num_cpu=num_cpu
        self.eval_episodes=eval_episodes
        self.train=train 
        self.test=test

    def run(self) -> None:
        
        ############ DQN models ############
        if self.model in ['DQN','dqn']:
            if self.train:
                self.DQN_train()
            if self.test:
                self.DQN_test()
        
        ############ PPO models ############
        elif self.model in ['PPO','ppo']:
            if self.train:
                self.PPO_train()
            if self.test:
                self.PPO_test()
        
        ############ A2C models ############
        elif self.model in ['A2C', 'a2c']:
            if self.train:
                self.A2C_train()
            if self.test:
                self.A2C_test()
        
        else:
            print(f'Model {self.model} not implemented.')
            return

    def PPO_train(self) -> None:
        env_args = {'image_mode':self.image_mode, 
                    'n_intruders' : self.n_intruders, 
                    'image_pixel_size' : self.image_size
                    }
        
        # Create the vectorised environments
        vec_env = make_vec_env('ConflictArt-v0', 
                               n_envs = self.num_cpu,
                               env_kwargs=env_args,
                               seed = self.seed)
        
        # Get the model
        model = PPO("CnnPolicy", vec_env, verbose = 1)
        
        # Train it
        model.learn(total_timesteps=int(3e6))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_ppo/model")
        del model
        
        # Close it
        vec_env.close()
        
    def PPO_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', render_mode=None)
        model = PPO.load(f"models/ConflictArt-v0_{self.image_mode}_ppo/model", env=env)

        for i in range(self.eval_episodes):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action[()])
        
        env.close()
        
    def A2C_train(self) -> None:
        # Create the vectorised environments
        vec_env = SubprocVecEnv([self.make_env('ConflictArt-v0', i) for i in range(self.num_cpu)])
        
        # Get the model
        model = A2C("CnnPolicy", vec_env, verbose = 1)
        
        # Train it
        model.learn(total_timesteps=int(3e6))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_a2c/model")
        del model
        
        # Close it
        vec_env.close()
        
    def A2C_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', render_mode=None)
        model = A2C.load(f"models/ConflictArt-v0_{self.image_mode}_a2c/model", env=env)

        for i in range(self.eval_episodes):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action[()])
        
        env.close()
        
        
        
    def DQN_train(self) -> None:
        # Create the environment
        env = gym.make('ConflictArt-v0', 
                    render_mode=None, 
                    n_intruders = self.n_intruders,
                    image_mode = self.image_mode,
                    image_pixel_size = self.image_size)
        
        # Set the random seed
        env.reset(self.seed)

        # Create the model
        model = DQN("CnnPolicy", env, verbose=1, 
                    buffer_size = 500_000,
                    optimize_memory_usage = True,
                    replay_buffer_kwargs={"handle_timeout_termination": False})

        # Train the model
        model.learn(total_timesteps=int(3e6))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_dqn/model")
        del model

        env.close()
        
    def DQN_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', render_mode=None)
        model = DQN.load(f"models/ConflictArt-v0_{self.image_mode}_dqn/model", env=env)

        for i in range(self.eval_episodes):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action[()])
        
        env.close()
    
    
if __name__ == "__main__":
    trainer = RLTrainer(model=RL_MODEL, 
                        image_mode=IMAGE_MODE, 
                        n_intruders = N_INTRUDERS, 
                        image_size = IMAGE_SIZE, 
                        seed = SEED, 
                        num_cpu = NUM_CPU, 
                        eval_episodes = EVAL_EPISODES, 
                        train = TRAIN, 
                        test = TEST)
    
    trainer.run()