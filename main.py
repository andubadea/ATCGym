import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import re
import atc_gym
import imageio.v2 as imageio

atc_gym.register_envs()
# check_env(gym.make('ConflictArt-v0', render_mode=None))

RL_MODEL = 'DQN'
IMAGE_MODE = 'rel_rgb'
N_INTRUDERS = 4
IMAGE_SIZE = 128
SEED = 42
NUM_CPU = 4
RENDER_MODE = None # None means no images, images means images
TRAIN_EPISODES = int(3e6)
EVAL_EPISODES = 1
TRAIN = True
TEST = False

class RLTrainer:
    def __init__(self, model:str = 'DQN', image_mode:str = 'rgb', n_intruders:int = 4, image_size:int = 128, 
                 seed:int = 42, num_cpu:int = 4, train_episodes:int = 1000, eval_episodes:int = 10, 
                 train:bool = True, test:bool = False, render_mode:str = None):
        self.model = model
        self.image_mode = image_mode
        self.n_intruders = n_intruders
        self.image_size = image_size
        self.seed = seed
        self.num_cpu = num_cpu
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.train = train 
        self.test = test
        self.render_mode = render_mode
        
        # Counter
        self.env_no = 0

    def run(self) -> None:
        # Delete existing images if any
        if self.test:
            # Attempt to delete all existing images
            to_delete = 'atc_gym/envs/debug/images/'
            for filename in os.listdir(to_delete):
                os.remove(to_delete + filename)
            
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
        
        if self.test and self.render_mode is not None:
            # Make the gif
            self.make_gif()

    def PPO_train(self) -> None:       
        # Create the vectorised environments
        vec_env = make_vec_env(self.make_env, 
                               n_envs = self.num_cpu,
                               vec_env_cls=SubprocVecEnv)
        
        # Get the model
        model = PPO("CnnPolicy", vec_env, verbose = 1)
        
        # Train it
        model.learn(total_timesteps=int(self.train_episodes))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_ppo/model")
        del model
        
        # Close it
        vec_env.close()
        
    def PPO_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', 
                render_mode=self.render_mode, 
                n_intruders = self.n_intruders,
                image_mode = self.image_mode,
                image_pixel_size = self.image_size)
        
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
        vec_env = make_vec_env(self.make_env, 
                               n_envs = self.num_cpu,
                               vec_env_cls=SubprocVecEnv)
        
        # Get the model
        model = A2C("CnnPolicy", vec_env, verbose = 1)
        
        # Train it
        model.learn(total_timesteps=int(self.train_episodes))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_a2c/model")
        del model
        
        # Close it
        vec_env.close()
        
    def A2C_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', 
                render_mode=self.render_mode, 
                n_intruders = self.n_intruders,
                image_mode = self.image_mode,
                image_pixel_size = self.image_size)
        
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
                    render_mode=self.render_mode, 
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
        model.learn(total_timesteps=int(self.train_episodes))
        
        # Save it
        model.save(f"models/ConflictArt-v0_{self.image_mode}_dqn/model")
        del model

        env.close()
        
    def DQN_test(self) -> None:
        #Test the trained model
        env = gym.make('ConflictArt-v0', 
                render_mode=self.render_mode, 
                n_intruders = self.n_intruders,
                image_mode = self.image_mode,
                image_pixel_size = self.image_size)
        
        model = DQN.load(f"models/ConflictArt-v0_{self.image_mode}_dqn/model", env=env)

        for i in range(self.eval_episodes):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Make steps
                obs, reward, done, truncated, info = env.step(action[()])
        
        env.close()
        
    def make_env(self):
        """
        Utility function for multiprocessed env.
        :param env_id: the environment ID
        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param rank: index of the subprocess
        """
        env = gym.make('ConflictArt-v0', 
                render_mode=self.render_mode, 
                n_intruders = self.n_intruders,
                image_mode = self.image_mode,
                image_pixel_size = self.image_size)

        env.reset(seed=self.seed + self.env_no)
        self.env_no +=1 
        return env
    
    def make_gif(self) -> None:
        # Get a list of all the images in the debug folder
        png_folder = 'atc_gym/envs/debug/images/'
        png_list = self.natural_sort([img for img in os.listdir(png_folder) if '.png' in img])
        # Create a gif
        images = []
        for img in png_list:
            images.append(imageio.imread(png_folder + img))
        imageio.mimsave('output/render.gif', images)
        
        # Clean up
        for filename in os.listdir(png_folder):
            os.remove(png_folder + filename)
                
    @staticmethod
    def natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
    
if __name__ == "__main__":
    trainer = RLTrainer(model=RL_MODEL, 
                        image_mode=IMAGE_MODE, 
                        n_intruders = N_INTRUDERS, 
                        image_size = IMAGE_SIZE, 
                        seed = SEED, 
                        num_cpu = NUM_CPU, 
                        train_episodes = TRAIN_EPISODES,
                        eval_episodes = EVAL_EPISODES, 
                        train = TRAIN, 
                        test = TEST,
                        render_mode = RENDER_MODE)
    
    trainer.run()