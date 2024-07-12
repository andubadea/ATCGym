import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import re
import atc_gym
import imageio.v2 as imageio
from typing import Any

atc_gym.register_envs()
#check_env(gym.make('ConflictSACArt-v0', render_mode=None))

ENV = 'ConflictSACArt-v0'
RL_MODEL = 'SAC'
IMAGE_MODE = 'rel_rgb'
DQN_BUFFER_SIZE = 500_000 # elements, needs tweaking if image is also tweaked
N_INTRUDERS = 4 # If none, then number of intruders is random every time
IMAGE_SIZE = 128
SEED = 42
NUM_CPU = 16
TRAIN_EPISODES = int(5e7)
EVAL_EPISODES = 10
RENDER_MODE = None # None means no images, images means images
TRAIN = True

class RLTrainer:
    def __init__(self, 
                 env:str, 
                 model:str = 'DQN', 
                 buffer_size:int = 500_000, 
                 image_mode:str = 'rgb', 
                 n_intruders:int = 4, 
                 image_size:int = 128, 
                 seed:int = 42, 
                 num_cpu:int = 4, 
                 train_episodes:int = 1000, 
                 eval_episodes:int = 10, 
                 train:bool = True,
                 render_mode:str = None) -> None:
        self.env = env
        self.model = model
        self.buffer_size = buffer_size
        self.image_mode = image_mode
        self.n_intruders = n_intruders
        self.image_size = image_size
        self.seed = seed
        self.num_cpu = num_cpu
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.train = train 
        self.render_mode = render_mode
        
        # Model save and load path
        self.model_path = f"models/{self.env}_{self.image_mode}_{n_intruders}_{self.model}/"
        
        # Env counter
        self.env_no = 0

    def run(self) -> None:
        # Get the model and the env
        model, env = self.get_model()
        if self.train:
            # Train then
            model.learn(total_timesteps=int(self.train_episodes))
            # Save final model
            model.save(self.model_path + "model")
            
        else:
            # Delete old gif files
            for filename in [x for x in os.listdir('output') if '.gif' in x]:
                os.remove('output/' + filename)
            # Do the eval
            for i in range(self.eval_episodes):
                done = truncated = False
                obs, info = env.reset()
                rew_sum = 0
                while not (done or truncated):
                    # Predict
                    action, _states = model.predict(obs, deterministic=True)
                    # Get reward
                    obs, reward, done, truncated, info = env.step(action[()])
                    rew_sum += reward
                    
                print(f'Episode: {i+1}/{self.eval_episodes} | total reward: {rew_sum}')
            
                # Make a gif
                if self.render_mode is not None:
                    self.make_gif(i)
        
        # Wrap up
        env.close()
        del model
            
    def get_model(self) -> Any:
        if self.model in ['DQN','dqn']:
            if self.train:
                # We train, make a vectorised environment
                env = make_vec_env(self.make_env, 
                                n_envs = self.num_cpu,
                                vec_env_cls=SubprocVecEnv)
                model = DQN("CnnPolicy", env, verbose = 1, 
                    buffer_size=self.buffer_size,
                    optimize_memory_usage=True,
                    replay_buffer_kwargs={"handle_timeout_termination":False})
                return model, env
            else:
                # Make a test environment
                env = gym.make(self.env, 
                            render_mode=self.render_mode, 
                            n_intruders = self.n_intruders,
                            image_mode = self.image_mode,
                            image_pixel_size = self.image_size)
                return DQN.load(self.model_path + "model", env=env), env
        
        ############ PPO models ############
        elif self.model in ['PPO','ppo']:
            if self.train:
                # We train, make a vectorised environment
                env = make_vec_env(self.make_env, 
                                n_envs = self.num_cpu,
                                vec_env_cls=SubprocVecEnv)
                model = PPO("CnnPolicy", env, verbose = 1)
                return model, env
            else:
                # Make a test environment
                env = gym.make(self.env, 
                            render_mode=self.render_mode, 
                            n_intruders = self.n_intruders,
                            image_mode = self.image_mode,
                            image_pixel_size = self.image_size)
                return PPO.load(self.model_path + "model", env=env), env
        
        ############ A2C models ############
        elif self.model in ['A2C', 'a2c']:
            if self.train:
                # We train, make a vectorised environment
                env = make_vec_env(self.make_env, 
                                n_envs = self.num_cpu,
                                vec_env_cls=SubprocVecEnv)
                model = A2C("CnnPolicy", env, verbose = 1)
                return model, env
            else:
                # Make a test environment
                env = gym.make(self.env, 
                            render_mode=self.render_mode, 
                            n_intruders = self.n_intruders,
                            image_mode = self.image_mode,
                            image_pixel_size = self.image_size)
                return A2C.load(self.model_path + "model", env=env), env
        
        ############ SAC models ############
        elif self.model in ['SAC', 'sac']:
            if self.train:
                # We train, make a vectorised environment
                env = make_vec_env(self.make_env, 
                                n_envs = self.num_cpu,
                                vec_env_cls=SubprocVecEnv)
                model = SAC("CnnPolicy", env, verbose = 1,
                    optimize_memory_usage=True,
                    replay_buffer_kwargs={"handle_timeout_termination":False})
                return model, env
            else:
                # Make a test environment
                env = gym.make(self.env, 
                            render_mode=self.render_mode, 
                            n_intruders = self.n_intruders,
                            image_mode = self.image_mode,
                            image_pixel_size = self.image_size)
                return SAC.load(self.model_path + "model", env=env), env
            
        else:
            print(f'Model {self.model} not implemented.')
            return None
        
    def make_env(self):
        """
        Utility function for multiprocessed env.
        :param env_id: the environment ID
        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param rank: index of the subprocess
        """
        env = gym.make(self.env, 
                render_mode=self.render_mode, 
                n_intruders = self.n_intruders,
                image_mode = self.image_mode,
                image_pixel_size = self.image_size)

        env.reset(seed=self.seed + self.env_no)
        self.env_no +=1 
        return env
    
    def make_gif(self) -> None:
        # Get a list of all the images in the debug folder
        png_folder = 'atc_gym/envs/data/images/'
        png_list = self.natural_sort([img for img in os.listdir(png_folder) 
                                            if '.png' in img])
        # Create a gif
        images = []
        for img in png_list:
            images.append(imageio.imread(png_folder + img))
        imageio.mimsave(f'output/Eval_{self.eval_no+1}.gif', 
                        images)
        
        # Clean up
        for filename in os.listdir(png_folder):
            os.remove(png_folder + filename)
        
        # Increment eval counter
        self.eval_no += 1
                
    @staticmethod
    def natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in 
                                                re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)     
        
    
if __name__ == "__main__":
    trainer = RLTrainer(env = ENV,
                        model = RL_MODEL, 
                        buffer_size = DQN_BUFFER_SIZE,
                        image_mode=IMAGE_MODE, 
                        n_intruders = N_INTRUDERS, 
                        image_size = IMAGE_SIZE, 
                        seed = SEED, 
                        num_cpu = NUM_CPU, 
                        train_episodes = TRAIN_EPISODES,
                        eval_episodes = EVAL_EPISODES, 
                        train = TRAIN, 
                        render_mode = RENDER_MODE)
    
    trainer.run()