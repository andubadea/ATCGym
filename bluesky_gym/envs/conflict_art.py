import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
matplotlib.rcParams['interactive'] = False
matplotlib.use('Agg')

class ConflictArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, n_intruders = 1):
        # Will want to eventually make these env properties
        self.n_intruders = 2 # number of intruders to spawn
        self.image_pixel_size = 300 # Resolution of image
        self.image_inch_size = 10 # Needed only for matplotlib
        self.playground_size = 100 # metres, also square
        self.min_travel_dist = self.playground_size/2 #metres, minimum travel distance
        self.rpz = 16 #metres, protection zone radius (minimum distance between two agents)
        self.mag_accel = 2 # m/s, constant acceleration magnitude
        self.max_speed = 18 #m/s, maximum speed both backwards and forwards
        self.default_speed = 10 #m/s, default speed for ownship
        
        # Useful
        self.n_ac = self.n_intruders + 1
        # Build observation space dict
        # Define it as an rgb image
        self.observation_space = spaces.Dict({
                "rgb_image": spaces.Box(low = 0, high = self.image_pixel_size, shape=(self.image_pixel_size,self.image_pixel_size,3), dtype=np.uint8)
        })
        
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
        info_dict["own_intrusions"] = self.intrusions
        return info_dict
        
    def reset(self, seed=None, options=None) -> list[np.ndarray, dict]:
        # This generates a new episode after one has been completed
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
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
                
        # Initialise number of intrusions
        self.intrusions = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        accel = self._action_to_accel[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
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
        
    
    def conflict_plot(self):
        fig = plt.figure(frameon=False)
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
        
        # Plot ownship info in green
        own_color = (0,1,0)
        plt.scatter(self.ac_locations[0][0],
                    self.ac_locations[0][1], 
                    marker='o', 
                    color = own_color,
                    s = 300) # Location
        
        plt.plot([self.ac_locations[0][0],
                  self.ac_targets[0][0]], 
                 [self.ac_locations[0][1], 
                 self.ac_targets[0][1]],
                 color = own_color) # Trajectory
        
        # Plot intruder info in red
        int_color = (1,0,0)
        for i in range(self.n_intruders):
            acidx = i + 1
            plt.scatter(self.ac_locations[acidx][0],
                        self.ac_locations[acidx][1], 
                        marker='o', 
                        color = int_color,
                        s = 300) # Location

            plt.plot([self.ac_locations[acidx][0],
                    self.ac_targets[acidx][0]], 
                    [self.ac_locations[acidx][1], 
                    self.ac_targets[acidx][1]],
                    color = int_color) # Trajectory

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
        fig.clear()
        plt.close()
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
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
    #env.conflict_plot()
    import timeit
    print(timeit.timeit('env.conflict_plot()', number = 1000, globals = globals())/1000)