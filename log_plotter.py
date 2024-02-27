import os
import matplotlib.pyplot as plt

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
            
    fig, ax = plt.subplots()
    ax.set_title(filename)
    ax.plot(timestep_no, rew_mean)
    ax.hlines(1, 0, max(timestep_no), color = 'red')
    ax.set_xlabel('Number of timesteps')
    ax.set_ylabel('Sum of rewards')
    plt.show()

logfiles = os.listdir('output')

for filename in logfiles:
    if '.log' in filename:
        plot_out_log(f'output/{filename}')

#scp sim6:~/Desktop/Andrei/BlueskyGym/\*.log  ./output/
