import os
import matplotlib.pyplot as plt
import numpy as np

moving_average_window = 1000

def plot_out_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    rew_mean = []
    timestep_no = []
        
    for line in lines:
        if 'ep_rew_mean' in line:
            split = line.split('|')
            rew = float(split[2])
            rew_mean.append(rew)
        if 'total_timesteps' in line:
            split = line.split('|')
            step = int(split[2])
            timestep_no.append(step)
    
    # Moving average
    mv_avg = []
    for i in range(moving_average_window, len(rew_mean)):
        mv_avg.append(np.average(rew_mean[i-moving_average_window:i]))
            
    fig, ax = plt.subplots()
    ax.set_title(f'{filename} mean reward')
    ax.plot(timestep_no, rew_mean)
    ax.hlines(1, 0, max(timestep_no), color = 'red')
    ax.set_xlabel('Number of timesteps')
    ax.set_ylabel('Sum of rewards')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title(f'{filename} moving average')
    ax.plot(range(len(mv_avg)), mv_avg)
    ax.hlines(1, 0, len(mv_avg), color = 'red')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Sum of rewards')
    plt.show()

logfiles = os.listdir('output')

for filename in logfiles:
    if 'out4.log' in filename:
        plot_out_log(f'output/{filename}')

#scp sim6:~/Desktop/Andrei/BlueskyGym/\*.log  ./output/
