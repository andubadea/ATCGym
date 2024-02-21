import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt

class ConflictArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, n_intruders = 1):
        # Will want to eventually make these env properties
        self.n_intruders = 1 # number of intruders to spawn
        self.image_size = 500 # pixels, square image
        self.playground_size = 100 # metres, also square
        self.min_travel_dist = 50 #metres, minimum travel distance
        self.rpz = 16 #metres, protection zone radius (minimum distance between two agents)
        self.mag_accel = 2 # m/s, constant acceleration magnitude
        self.max_speed = 18 #m/s, maximum speed both backwards and forwards
        self.default_speed = 10 #m/s, default speed for ownship
        
        # Useful
        self.n_ac = self.n_intruders + 1
        # Build observation space dict
        # Define it as an rgb image
        self.observation_space = spaces.Dict({
                "rgb_image": spaces.Box(low = 0, high = self.image_size, shape=(self.image_size,self.image_size,3), dtype=np.uint8)
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
        
    def _get_obs(self):
        # Here we basically create the image
        # Matplotlib should work
        fig, ax = plt.subplots()
        # Set the axis limits
        ax.set_xlim([0, self.playground_size])
        ax.set_ylim([0, self.playground_size])
        # Plot ownship info in green
        plt.scatter(self.ac_locations[0][0],self.ac_locations[0][1], marker='o', color = 'green') # Location
        plt.scatter(self.ac_targets[0][0], self.ac_targets[0][1], marker='x', color = 'green')
        
        
        pass
    
    def _get_info(self):
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
        
    def reset(self, seed=None, options=None):
        # This generates a new episode after one has been completed
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Initialise all aircraft. Ownship will always be the one with index 0, the rest are intruders
        # Ensure the initial locations are spaced enough apart
        self.ac_locations = generate_points_with_min_distance(self.n_ac, 
                                                              (self.playground_size,self.playground_size),
                                                              self.rpz*2)
        self.ac_targets = self.ac_locations # Initialise targets
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
        direction = self._action_to_direction[action]
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
        return np.linalg.norm(self.ac_locations[acidx] - self.ac_targets[acidx], ord=1)
    
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
            
            
# Some helper functions
# Credit: @ch271828n and @Samir
# https://stackoverflow.com/a/57353855
def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                                high=max_movement,
                                size=(len(coords), 2))
    coords += noise

    return coords