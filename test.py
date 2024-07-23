import imageio.v2 as imageio
import re
import numpy as np
import os
from atc_gym.envs.conflict_art import ConflictArtEnv
from atc_gym.envs.conflict_gen_art import ConflictGenArtEnv
from atc_gym.envs.conflict_urban_art import ConflictUrbanArtEnv
from atc_gym.envs.conflict_multi_art import ConflictMultiArtEnv

def natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
    
def make_gif() -> None:
        # Get a list of all the images in the debug folder
        png_folder = f'{os.path.dirname(__file__)}atc_gym/envs/data/images/'
        png_list = natural_sort([img for img in os.listdir(png_folder) if '.png' in img])
        # Create a gif
        images = []
        for img in png_list:
            images.append(imageio.imread(png_folder + img))
        imageio.mimsave('output/render.gif', images)
        
        # Clean up
        for filename in os.listdir(png_folder):
            os.remove(png_folder + filename)
        
# Testing
if __name__ == "__main__":
    # Variables
    n_intruders = 3
    image_mode = 'rgb'
    image_pixel_size = 128
    
    # Make environment
    env = ConflictMultiArtEnv('images', n_intruders, image_mode, image_pixel_size)
    env.reset()
    
    #Test images
    if False:
        done = truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(0)
        make_gif()
    
    #Test env creation
    if False:
        for a in range(100):
            env.reset()
            env.step(0)
    
    #Test average dumb reward
    if True:
        rolling_avg = []
        rew_list = []
        tests_num = 200
        for a in range(tests_num):
            env.reset()
            rew_sum = 0
            steps = 0
            done = truncated = False
            while not (done or truncated):
                spd = np.array([[0],[0],[0]])
                obs, reward, done, truncated, info = env.step(spd)
                rew_sum += reward
                steps += 1
                
            rew_list.append(rew_sum)
            rolling_avg.append(np.average(rew_list))
                
            print(f'Episode: {a+1}/{tests_num} | avg: {rew_sum} | rolling avg: {np.average(rolling_avg)} | steps: {steps}')