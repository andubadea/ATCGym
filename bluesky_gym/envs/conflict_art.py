import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple, Dict, Any, List
matplotlib.rcParams['interactive'] = False
matplotlib.use('Agg')

class ConflictArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, n_intruders = 1):
        # Will want to eventually make these env properties
        self.n_intruders = 9 # number of intruders to spawn
        self.playground_size = 100 # metres, also square
        self.min_travel_dist = self.playground_size/2 #metres, minimum travel distance
        self.rpz = 10 #metres, protection zone radius (minimum distance between two agents)
        self.mag_accel = 2 # m/s, constant acceleration magnitude
        self.max_speed = 15 #m/s, maximum speed both backwards and forwards
        self.default_speed = 10 #m/s, default speed for ownship
        
        # Image properties
        self.image_pixel_size = 128 # Resolution of image
        self.image_inch_size = 10 # Needed only for matplotlib
        
        # Simulation properties
        self.dt = 0.1 # seconds, simulation time step
        self.action_dt = 1 #seconds, action time step
        self.step_no = 0 #sim step counter
        self.max_steps = 500 #maximum steps per episode
        
        # Useful calculated properties
        self.n_ac = self.n_intruders + 1
        self.target_tolerance = self.max_speed * self.dt * 1.1 # to make sure that this condition is met
        
        # Debugging mode
        self.debug = False
        
        # Build observation space dict, define it as an rgb image
        self.observation_space = spaces.Box(low = 0, high = 255, shape=(self.image_pixel_size,self.image_pixel_size,3), dtype=np.uint8)
        
        # 3 actions: Nothing, Accelerate, Decelerate
        self.action_space = spaces.Discrete(3)
        
        # To map the actions to accelerations, we need to define a constant acceleration
        self._action_to_accel = {
            0:0, # Maintain speed            
            1:self.mag_accel, #accelerate
            2:-self.mag_accel, #decelerate
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None
        
    def _get_obs(self) -> np.ndarray:
        # Here we basically create the image
        return self.conflict_plot()
    
    def _get_info(self) -> dict:
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # Initialise info dict
        info_dict = {}
        # Ownship stuff
        info_dict["own_location"] = self.ac_locations[0]
        info_dict["own_target"] = self.ac_targets[0]
        info_dict["own_dist"] = self.dist2target(0)
        # Intruder infos
        for i in range(self.n_intruders):
            acidx = i + 1
            info_dict[f"int{acidx}_location"] = self.ac_locations[acidx]
            info_dict[f"int{acidx}_target"] = self.ac_targets[acidx]
            info_dict[f"int{acidx}_dist"] = self.dist2target(acidx)
        # Intrusions
        info_dict["own_intrusion_steps"] = self.intrusion_time_steps
        return info_dict
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        # This generates a new episode after one has been completed
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.step_no = 0
        
        # Initialise all aircraft. Ownship will always be the one with index 0, the rest are intruders
        # Ensure the initial locations are spaced enough apart
        self.ac_locations = self.generate_origins()
        self.ac_targets = np.copy(self.ac_locations) # Initialise targets
        # Random intruder AC speeds between 50% and 90% of the maximum speed
        self.ac_speeds = self.np_random.uniform(0.5, 0.9, self.n_ac) * self.max_speed
        # Set the initial speed of the ownship as the default speed
        self.ac_speeds[0] = self.default_speed

        # We will sample the target locations randomly until it is far enough from initial location
        for acidx in range(self.n_ac):
            # Distance to target
            while self.dist2target(acidx) < self.min_travel_dist:
                self.ac_targets[acidx] = self.np_random.random(2) * self.playground_size
        
        # For intruders, we want the targets to be very far away so they leave the screen eventually
        self.scale_intruder_targets()
        
        # Number of intrusion time steps
        self.intrusion_time_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to ownship acceleration
        accel = self._action_to_accel[action]
        # Update the velocity of the ownship in function of this acceleration
        self.ac_speeds[0] += accel * self.dt
        # If the speed is above or below the maximum speed, cap it
        self.ac_speeds[0] = np.clip(self.ac_speeds[0], -self.max_speed, self.max_speed)
        # Update the positions of all aircraft
        self.update_pos()
        
        # Get distance of the ownship to the target
        own_dist2goal = self.dist2target(0)
        # Get the distances of the ownship to other aircraft
        own_dist2others = self.dist2others(0)[1:] # skip distance to self, it's 0
        # Agent is successful if the target is reached
        success = own_dist2goal < self.target_tolerance
        # An intrusion occurs when any distance to others is smaller than the protection zone
        intrusion = np.any(own_dist2others < self.rpz)
        self.intrusion_time_steps += 1 if intrusion else 0
        # We terminate the episode if the ownship is successful or too many time steps have passed
        terminated = success or self.step_no > self.max_steps
        
        # We reward success and penalise intrusions
        reward = 1 if success else 0 # reward for finishing
        reward = reward if not intrusion else reward - 0.1 # intrusions are penalised
        
        # Get needed info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        if self.debug:
            print(f'\n----- Step {self.step_no} -----')
            print(f'Distance to target: {own_dist2goal}')
            print(f'Distance to others: {own_dist2others}')
            print(f'Acceleration: {accel}')
            print(f'Speed: {self.ac_speeds[0]}')
            print(f'Intrusion: {intrusion}')
            print(f'Reward: {reward}')
            print(f'Terminated: {terminated}')
        
        self.step_no += 1
        
        return observation, reward, terminated, False, info
    
    def scale_intruder_targets(self) -> None:
        # Get target directions
        dir_v = self.ac_targets - self.ac_locations
        # Get unit vectors
        unit_dir_v = np.transpose(dir_v.T / np.linalg.norm(dir_v, axis = 1))
        # Extend the target by a lot
        new_targets = self.ac_locations + unit_dir_v * self.playground_size * 10
        # Set the intruder targets as this
        self.ac_targets[1:] = new_targets[1:]
        return
        
    
    def update_pos(self) -> None:
        # Get the direction vectors for all aircraft directions
        dir_v = self.ac_targets - self.ac_locations
        # Get unit vector
        unit_dir_v = np.transpose(dir_v.T / np.linalg.norm(dir_v, axis = 1))
        # Now update position
        self.ac_locations += np.transpose((self.ac_speeds * self.dt) * unit_dir_v.T)
        return
        
    def dist2target(self, acidx:int) -> float:
        """Returns the distance from the current location to the target
        for aircraft acidx.

        Args:
            acidx (int): Aircraft index

        Returns:
            float: Distance to target in metres.
        """
        return np.linalg.norm(self.ac_locations[acidx] - self.ac_targets[acidx])
    
    def dist2others(self, acidx:int) -> np.ndarray:
        """Returns an array of the distance from this aircraft to
        all other aircraft.

        Args:
            acidx (int): Aircraft index

        Returns:
            np.ndarray: Distance array (own index is 0)
        """
        return np.linalg.norm(self.ac_locations[acidx] - self.ac_locations, axis=1)
    
    def dist2targets(self) -> np.ndarray:
        """Returns the distances to all targets

        Returns:
            np.ndarray: Array of distances in metres.
        """
        return np.linalg.norm(self.ac_locations - self.ac_targets, axis = 1)
        
    
    def conflict_plot(self) -> np.ndarray:
        fig = plt.figure()
        fig.set_size_inches(self.image_inch_size,self.image_inch_size)
        fig.set_dpi(self.image_pixel_size / self.image_inch_size)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # Set the axis limits
        ax.set_xlim([0, self.playground_size])
        ax.set_ylim([0, self.playground_size])
        # Ratio
        ax.set_aspect('equal')
        # Get rid of axes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_axis_off()
        # Get rid of white space
        plt.tight_layout(pad=-5)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # Encode the velocity of the ownship and intruders within the blue channel
        # 0 means maximum negative speed, 1 means maximum positive speed
        int_color = (1,0,0) # Default red, used to trajectory lines
        int_spd_color = np.zeros((self.n_intruders, 3))
        int_spd_color[:,0] += 1 # By default intruders are red
        int_spd_color[:,2] += self.ac_speeds[1:]/self.max_speed # Encode the speed in the blue channel
    
        # Plot intruder info in red
        for i in range(self.n_intruders):
            acidx = i + 1
            ax.scatter(self.ac_locations[acidx][0],
                        self.ac_locations[acidx][1], 
                        marker='o', 
                        color = int_spd_color[i],
                        s = 600) # Location

            ax.plot([self.ac_locations[acidx][0],
                    self.ac_targets[acidx][0]], 
                    [self.ac_locations[acidx][1], 
                    self.ac_targets[acidx][1]],
                    color = int_color,
                    linewidth=2) # Trajectory
            
        # Plot ownship info in green
        # Encode its speed in the blue channel
        own_color = (0,1,0)
        own_spd_color = (0,1,self.ac_speeds[0]/self.max_speed)
        ax.scatter(self.ac_locations[0][0],
                    self.ac_locations[0][1], 
                    marker='o', 
                    color = own_spd_color,
                    s = 600,
                    linewidths=3) # Location
        
        ax.plot([self.ac_locations[0][0],
                  self.ac_targets[0][0]], 
                 [self.ac_locations[0][1], 
                 self.ac_targets[0][1]],
                 color = own_color,
                 linewidth=2) # Trajectory
        
        ax.scatter(self.ac_targets[0][0],
                   self.ac_targets[0][1],
                   marker = 'x',
                   color = own_color,
                   s = 300,
                   linewidths=5)

        # Add ax to figure 
        fig.add_axes(ax)
        # Get figure as numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()  # update/draw the elements
        _, _, w, h = canvas.figure.bbox.bounds
        w, h = int(w), int(h)
        buf = canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        rgb_array = np.array(image.reshape(h, w, 4))
        # For some reason we get a gray border, remove it
        rgb_array[0, :] = np.zeros((self.image_pixel_size, 4)) + 255
        rgb_array[self.image_pixel_size-1,:] = np.zeros((self.image_pixel_size, 4)) + 255
        rgb_array[:, 0] = np.zeros((self.image_pixel_size, 4)) + 255
        rgb_array[:, self.image_pixel_size-1] = np.zeros((self.image_pixel_size, 4)) + 255
        # Clear memory
        fig.clear()
        plt.close()
        if self.debug:
            fig_debug, ax = plt.subplots()
            ax.imshow(rgb_array)
            fig_debug.savefig(f'./debug/images/{self.step_no}.png')
            fig_debug.clear()
            plt.close()
        return rgb_array[:,:,:3]
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # TODO: This
        pass
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
    def generate_origins(self) -> np.ndarray:
        """Generates aircraft origins with a minimum distance between them.

        Args:
            n (_type_): number of aircraft
            shape (_type_): dimensionality (2d,3d, etc)
            min_dist (_type_): minimum distance (metres)

        Returns:
            _type_: ndarray
        """
        # TODO: If needed, make it not be O(n2)
        # Initialise coords array
        coords = np.zeros((self.n_ac, 2))
        # First starting point is random, set all as this point
        coords += self.np_random.random(2) * self.playground_size
        # Now set subsequent points such that distance is met
        for i in range(self.n_intruders):
            acidx = i + 1
            # Get distance to previous origins
            dist = np.linalg.norm(coords[acidx] - coords[:acidx], axis=1)
            while np.any(dist < self.rpz*2):
                # Set random coords
                coords[acidx] = self.np_random.random(2) * self.playground_size
                dist = np.linalg.norm(coords[acidx] - coords[:acidx], axis=1)
        return coords

# Testing
if __name__ == "__main__":
    env = ConflictArtEnv()
    env.reset()
    for a in range(200):
        env.step(1)
    #env.conflict_plot()
    # import timeit
    # print(timeit.timeit('env.step(0)', number = 500, globals = globals())/500)