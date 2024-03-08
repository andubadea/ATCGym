import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import os
from shapely.geometry import Polygon, LineString
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple, Dict, Any, List
matplotlib.rcParams['interactive'] = False
matplotlib.use('Agg')

class ConflictUrbanArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    metadata = {"render_modes": ["images"], 
                "render_fps": 4, 
                "image_mode": ["rgb", "rel_rgb", "rel_gry"]}
    def __init__(self, render_mode=None, n_intruders = 1, image_mode = 'rgb', image_pixel_size = 128):
        # Will want to eventually make these env properties
        self.n_intruders = n_intruders # number of intruders to spawn
        self.playground_size = 100 # metres, also square
        self.min_travel_dist = 60 #metres, minimum travel distance
        self.rpz = 10 #metres, protection zone radius (minimum distance between two agents)
        self.mag_accel = 2 # m/s, constant acceleration magnitude
        self.max_speed = 15 #m/s, maximum speed
        self.default_speed = 5 #m/s, starting speed for ownship
        
        # Center square properties
        self.cntr_sq = 0.3 # Ratio of centre square size to env size
        self.cntr_sq_size = self.cntr_sq * self.playground_size
        self.cntr_low_coord = (1-self.cntr_sq)/2 * self.playground_size # Lower coord for both x and y
        self.cntr_high_coord = ((1-self.cntr_sq)/2 + self.cntr_sq) * self.playground_size # Lower coord
        self.cntr_poly = Polygon([[self.cntr_low_coord, self.cntr_low_coord], #bottom left
                                  [self.cntr_high_coord, self.cntr_low_coord], #bottom right
                                  [self.cntr_high_coord, self.cntr_high_coord], # top right
                                  [self.cntr_low_coord, self.cntr_high_coord]]) #top left
        
        # Route properties
        self.wp_res = 10 #metres, waypoint resolution
        self.wp_res_var = 5 # Variance allowed in the resolution
        self.hdg_range = 25 #deg, heading range where the next waypoint can be w.r.t. dest
        self.clip_tol = 1 #metres, tolerance at which a waypoint just clips to another existing one
        self.dest_tol = 15 #meteres, distance at which route clips to destination
        self.turn_prob = 0.10 # chance that a wp is big turn
        
        # Image properties
        self.image_pixel_size = image_pixel_size # Resolution of image
        self.image_inch_size = 10 # Needed only for matplotlib, don't change
        
        # Simulation properties
        self.dt = 0.1 # seconds, simulation time step
        self.action_dt = 1 #seconds, action time step
        self.step_no = 0 #sim step counter
        self.max_steps = 500 #maximum steps per episode
        
        # Useful calculated properties
        self.n_ac = self.n_intruders + 1
        self.target_tolerance = self.max_speed * self.dt * 1.1 # to make sure that this condition is met
        
        # Initialisations, should be defined in reset
        self.ac_targets = None
        self.step_no = None
        self.ac_locations = None
        self.ac_routes = None
        self.ac_wpidx = None
        self.ac_current_wp = None
        self.ac_speeds = None
        self.intrusion_time_steps = None
        
        assert image_mode in self.metadata["image_mode"]
        self.image_mode = image_mode
        
        # Build observation space dict, define it as an rgb image
        if image_mode == 'rgb' or image_mode == 'rel_rgb':
            self.observation_space = spaces.Box(low = 0, high = 255, shape=(self.image_pixel_size,self.image_pixel_size,3), dtype=np.uint8)
        elif image_mode == 'rel_gry':
            self.observation_space = spaces.Box(low = 0, high = 255, shape=(self.image_pixel_size,self.image_pixel_size,1), dtype=np.uint8)
        
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
        self.ac_routes = self.generate_routes()
        self.ac_wpidx = np.ones(self.n_ac, dtype=int) # waypoint index
        self.ac_current_wp = np.array([route[1] for route in self.ac_routes]) # target is first wp
        # Random intruder AC speeds between 50% and 90% of the maximum speed
        self.ac_speeds = self.np_random.uniform(0.5, 0.9, self.n_ac) * self.max_speed
        # Set the initial speed of the ownship as the default speed
        self.ac_speeds[0] = self.default_speed
        
        # Number of intrusion time steps
        self.intrusion_time_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action) -> Any:
        # Map the action (element of {0,1,2,3}) to ownship acceleration
        accel = self._action_to_accel[action]
        # Update the velocity of the ownship in function of this acceleration
        self.ac_speeds[0] += accel * self.dt
        # If the speed is above or below 0, cap it
        self.ac_speeds[0] = np.clip(self.ac_speeds[0], 0, self.max_speed)
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
        
        # if self.debug:
        #     print(f'\n----- Step {self.step_no} -----')
        #     print(f'Distance to target: {own_dist2goal}')
        #     print(f'Distance to others: {own_dist2others}')
        #     print(f'Acceleration: {accel}')
        #     print(f'Speed: {self.ac_speeds[0]}')
        #     print(f'Intrusion: {intrusion}')
        #     print(f'Reward: {reward}')
        #     print(f'Terminated: {terminated}')
        
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
        # Get distances to current waypoints
        dist_wps = self.dist2curwps()
        # Aircraft that reached their wp switch to the next one
        for acidx in np.argwhere(dist_wps < self.target_tolerance).flatten():
            self.ac_wpidx[acidx] += 1
            # Only update waypoint if it's not the last waypoint
            if self.ac_wpidx[acidx] < len(self.ac_routes[acidx]):
                self.ac_current_wp[acidx] = self.ac_routes[acidx][self.ac_wpidx[acidx]]
        
        # Get the direction vectors for all aircraft directions
        dir_v = self.ac_current_wp - self.ac_locations
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
    
    def dist2curwps(self) -> np.ndarray:
        """Returns the distances to all current waypoints

        Returns:
            np.ndarray: Array of distances in metres.
        """
        return np.linalg.norm(self.ac_locations - self.ac_current_wp, axis = 1)
    
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
    
    def dist2points(self, point, points) -> np.ndarray:
        """General distance from one point to other points.

        Returns:
            np.ndarray: distances
        """
        return np.linalg.norm(point - points, axis = 1)
        
    
    def conflict_plot(self) -> np.ndarray:
        fig = plt.figure()
        fig.set_size_inches(self.image_inch_size,self.image_inch_size)
        fig.set_dpi(self.image_pixel_size / self.image_inch_size)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # Set the axis limits
        if 'rel' in self.image_mode:
            # Relative images
            ax.set_xlim([self.ac_locations[0][0] - self.playground_size / 2, 
                     self.ac_locations[0][0] + self.playground_size / 2])
            ax.set_ylim([self.ac_locations[0][1] - self.playground_size / 2, 
                        self.ac_locations[0][1] + self.playground_size / 2])
        else:
            # Normal images
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
            # Get route
            route = self.ac_routes[acidx][self.ac_wpidx[acidx]:]
            # Insert current location
            route.insert(0, self.ac_locations[acidx])
            # np arr
            route_arr = np.array(route)
            ax.plot(route_arr[:,0], route_arr[:,1],
                    color = int_color,
                    linewidth=2) # Trajectory
            
        # Plot ownship info in green
        # Encode its speed in the blue channel
        own_color = (0,1,0)
        own_spd_color = (0,1, self.ac_speeds[0]/self.max_speed)
        ax.scatter(self.ac_locations[0][0],
                    self.ac_locations[0][1], 
                    marker='o', 
                    color = own_spd_color,
                    s = 600,
                    linewidths=3) # Location
        # Get route
        route = self.ac_routes[0][self.ac_wpidx[0]:]
        # Insert current location
        route.insert(0, self.ac_locations[0])
        # np arr
        route_arr = np.array(route)
        ax.plot(route_arr[:,0], route_arr[:,1],
                 color = own_color,
                 linewidth=2) # Trajectory
        
        # Plot the targets
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
        # Invert all the colors, 255 becomes 0
        rgb_array = np.abs(rgb_array-255)
        if 'gry' in self.image_mode:
            plot_array = self.rgb2gray(rgb_array[:,:,:3]).reshape((self.image_pixel_size, self.image_pixel_size, 1)).astype(np.uint8)
        elif 'rgb' in self.image_mode:
            plot_array = rgb_array[:,:,:3]
        else:
            plot_array = rgb_array
        # Clear memory
        fig.clear()
        plt.close()
        if self.render_mode == "images":
            fig_debug, ax = plt.subplots()
            ax.imshow(plot_array)
            dirname = os.path.dirname(__file__)
            fig_debug.savefig(f'{dirname}/debug/images/{self.step_no}.png')
            fig_debug.clear()
            plt.close()
        return plot_array
        
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
    def generate_origins(self) -> np.ndarray:
        """Generates aircraft origins with a minimum distance between them.
        The origins are also outside of the centre.

        Args:
            n (_type_): number of aircraft
            shape (_type_): dimensionality (2d,3d, etc)
            min_dist (_type_): minimum distance (metres)

        Returns:
            _type_: ndarray
        """
        # Initialise coords array
        coords = np.zeros((self.n_ac, 2))
        # First starting point is random, set all as this point
        coords += self.np_random.random(2) * self.playground_size
        # Keep setting points until we're outside centre
        inside_centre = np.all((self.cntr_low_coord<coords[0]) & 
                                   (coords[0]<self.cntr_high_coord))
        while inside_centre:
            coords = np.zeros((self.n_ac, 2))+ self.np_random.random(2) * self.playground_size
            inside_centre = np.all((self.cntr_low_coord<coords[0]) & 
                                   (coords[0]<self.cntr_high_coord))
        # Now set subsequent points such that distance is met
        for i in range(self.n_intruders):
            acidx = i + 1
            # Get distance to previous origins
            dist = np.linalg.norm(coords[acidx] - coords[:acidx], axis=1)
            # Also check if point is in the middle
            inside_centre = np.all((self.cntr_low_coord<coords[acidx]) & 
                                   (coords[acidx]<self.cntr_high_coord))
            # Keep going till it clicks
            while np.any(dist < self.rpz*3) or inside_centre:
                # Set random coords
                coords[acidx] = self.np_random.random(2) * self.playground_size
                dist = np.linalg.norm(coords[acidx] - coords[:acidx], axis=1)
                inside_centre = np.all((self.cntr_low_coord<coords[acidx]) & 
                                       (coords[acidx]<self.cntr_high_coord))
        return coords
    
    def generate_routes(self) -> List:
        """Generates aircraft routes that mimmic streets.

        Returns:
            np.ndarray: _description_
        """
        # We have origins we must now generate an intersection point.
        # It needs to be somewhere in the centre of the playground, so let's place it in 
        # a square surrounding the centre with its side = 1/4 playgound size
        intersection = self.np_random.random(2) * self.cntr_sq_size + self.cntr_low_coord
        # Next, we want to generate the targets such that a line connecting it and the origin
        # passes through the centre square
        self.ac_targets = self.np_random.random((self.n_ac, 2)) * self.playground_size # Initialise targets
        # We will sample the target locations randomly until it is far enough from origin
        # It also needs to be outside the centre square
        # The line connecting origin and destination must also intersect the centre square
        for acidx in range(self.n_ac):
            dist_ok = self.dist2target(acidx) > self.min_travel_dist # distance from origin
            # Make sure it is directed towards the centre
            sq_intersects = self.cntr_poly.intersects(LineString([self.ac_locations[acidx],
                                                                  self.ac_targets[acidx]]))
            # Make sure it's outside centre
            inside_centre = np.all((self.cntr_low_coord<self.ac_targets[acidx]) & 
                                   (self.ac_targets[acidx]<self.cntr_high_coord))
            
            while not (dist_ok and sq_intersects and not inside_centre):
                # Try again
                self.ac_targets[acidx] = self.np_random.random(2) * self.playground_size
                # Do the checks
                dist_ok = self.dist2target(acidx) > self.min_travel_dist
                sq_intersects = self.cntr_poly.intersects(LineString([self.ac_locations[acidx],
                                                                  self.ac_targets[acidx]]))
                inside_centre = np.all((self.cntr_low_coord<self.ac_targets[acidx]) & 
                                       (self.ac_targets[acidx]<self.cntr_high_coord))
        
        # For intruders, we want the targets to be very far away so they leave the screen eventually
        self.scale_intruder_targets()
        
        # Initialise waypoint database
        waypoint_database = []
        # Also initialise aircraft waypoint list
        ac_routes = []
        
        # Now, for each aircraft, we want to generate waypoints that kind of look like a street
        for acidx in range(self.n_ac):
            at_dest = False
            waypoints = [list(self.ac_locations[acidx])] # First waypoint is the origin
            while not at_dest:
                # Get a new pseudo-random waypoint
                new_wp = self.create_new_wp(waypoints[-1], self.ac_targets[acidx], acidx)
                # Check the distances to all the other waypoints in the database
                if acidx > 0: # Database empty for ownship so skip
                    dist_to_wp = self.dist2points(new_wp, waypoint_database)
                    # Any waypoint closer than the tolerance? New waypoint becomes
                    # the one with the lowest distance
                    min_idx = np.argmin(dist_to_wp)
                    if dist_to_wp[min_idx] < self.clip_tol:
                        new_wp = np.array(waypoint_database[min_idx])
                
                # If the new wp is within the center box, make the new wp as the intersection
                if (list(intersection) not in waypoints) and \
                            np.all((self.cntr_low_coord<new_wp) & 
                                    (new_wp<self.cntr_high_coord)):
                    new_wp = intersection
                        
                # Also clip if we are within range of the target or we have too many waypoints
                if np.linalg.norm(self.ac_targets[acidx] - new_wp) < self.dest_tol or \
                    len(waypoints) > 20:
                    new_wp = self.ac_targets[acidx]
                    at_dest = True
                
                # Append new wp to waypoints
                waypoints.append(list(new_wp))
            
            # Append the route to the list of routes
            ac_routes.append(waypoints)
            # Now take out intersection from the route if it is in it
            if (list(intersection) in waypoints):
                waypoints.pop(waypoints.index(list(intersection)))
            # Add the waypoints of this route to the database
            waypoint_database += waypoints

        # return the routes
        return ac_routes
                
    def create_new_wp(self, current_wp, target, acidx) -> np.ndarray:
        while True:
            # Get target direction
            dir_v = target - current_wp
            # Get unit vector
            unit_dir_v = np.transpose(dir_v / np.linalg.norm(dir_v))
            # Select a random angle within range to rotate it by
            # Random chance that the angle is between hdg range and 90
            big_turn = self.np_random.random() < self.turn_prob
            if big_turn:
                rot_angle = self.np_random.integers(45, 90) * self.np_random.choice([-1,1])
            else:
                rot_angle = self.np_random.integers(-self.hdg_range, self.hdg_range)
            # Rotate this vector
            new_dir_v = self.rotate_vector(unit_dir_v, rot_angle)
            # Get a random distance to extend it by
            if big_turn:
                # Give this a bigger distance
                distance = self.np_random.integers(-self.wp_res_var,self.wp_res_var) + self.wp_res * 2
            else:
                distance = self.np_random.integers(-self.wp_res_var,self.wp_res_var) + self.wp_res
            # Now extend it
            new_wp = current_wp + new_dir_v * distance
            # Check if it is within playground bounds, but only for the ownship
            if np.all((0 < new_wp) & (new_wp < self.playground_size)):
                return new_wp
            elif acidx > 0:
                # Outside playground but this is an intruder, so just set the next wp as the target
                return np.array(target)
            else:
                # Try again
                continue
    
    # Some helper functions
    @staticmethod
    def rgb2gray(rgb) -> np.ndarray:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    @staticmethod
    def rotate_vector(vec, angle):
        angle = np.deg2rad(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c,-s), (s, c)))
        return np.dot(R, vec)
        
# Testing
if __name__ == "__main__":
    # Variables
    n_intruders = 4
    image_mode = 'rel_rgb'
    image_pixel_size = 128
    
    # Make environment
    env = ConflictUrbanArtEnv('images', n_intruders, image_mode, image_pixel_size)
    env.reset()
    
    #Test images
    # done = truncated = False
    # while not (done or truncated):
    #     obs, reward, done, truncated, info = env.step(0)
    
    # Test env creation
    #env.step(0)
    
    # Test step time
    # import timeit
    # print(timeit.timeit('env.step(0)', number = 500, globals = globals())/500)
    
    # Test average dumb reward
    # rolling_avg = []
    # rew_list = []
    # tests_num = 100
    # for a in range(tests_num):
    #     print(f'Episode: {a+1}/{tests_num} | rolling avg: {np.average(rolling_avg)}')
    #     env.reset()
    #     rew_sum = 0
    #     done = truncated = False
    #     while not (done or truncated):
    #         obs, reward, done, truncated, info = env.step(0)
    #         rew_sum += reward
            
    #     rew_list.append(rew_sum)
    #     rolling_avg.append(np.average(rew_list))
    #     while len(rolling_avg) > 100:
    #         rolling_avg.pop(0)
    # plt.figure()
    # plt.plot(range(len(rolling_avg)),rolling_avg)
    # plt.savefig('hi.png')