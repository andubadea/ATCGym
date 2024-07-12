import numpy as np
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
import os
import osmnx as ox
import networkx as nx
from shapely.ops import linemerge
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple, Dict, Any, List
matplotlib.rcParams['interactive'] = False
matplotlib.use('Agg')

class ConflictSACArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    
    metadata = {"render_modes": ["images"], 
                "render_fps": 4, 
                "image_mode": ["rgb", "rel_rgb", "rel_gry"]}
    
    def __init__(self, render_mode=None, n_intruders = None, 
                                image_mode = 'rgb', image_pixel_size = 128):
        # Will want to eventually make these env properties
        self.intr_max = 5
        if n_intruders is None:
            self.intruders_random = True
            self.n_intruders = self.np_random.integers(1, self.intr_max+1)
        else:
            self.intruders_random = False
            self.n_intruders = n_intruders # number of intruders to spawn
        self.playground_size = 200 # metres, also square
        self.min_travel_dist = 100 #metres, minimum travel distance
        self.rpz = 16 #metres, protection zone radius (minimum distance between two agents)
        self.mag_accel = 3.5 # m/s, constant acceleration magnitude
        self.max_speed = 18 #m/s, maximum speed
        self.default_speed = 8 #m/s, starting speed for ownship
        self.max_speed_diff = self.max_speed - self.default_speed
        
        # Load the city graph
        self.G, self.nodes_m, self.edges_m = self.load_graph('Vienna')
        
        # Image properties
        self.image_pixel_size = image_pixel_size # Resolution of image
        self.image_inch_size = 10 # Needed only for matplotlib, don't change
        
        # Simulation properties
        self.dt = 0.1 # seconds, simulation time step
        self.step_no = 0 #sim step counter
        self.max_steps = 1000 #maximum steps per episode
        
        # Useful calculated properties
        self.n_ac = self.n_intruders + 1
        # to make sure that this condition is met
        self.target_tolerance = self.max_speed * self.dt * 1.1
        
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
            self.observation_space = spaces.Box(low = 0, high = 255, 
                                        shape=(self.image_pixel_size,
                                               self.image_pixel_size,3), 
                                        dtype=np.uint8)
        elif image_mode == 'rel_gry':
            self.observation_space = spaces.Box(low = 0, high = 255, 
                                        shape=(self.image_pixel_size,
                                               self.image_pixel_size,1), 
                                        dtype=np.uint8)
        
        # 0 = default speed, -1 = min speed, 1 = max speed
        self.action_space = spaces.Box(-1,1)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "images":
            to_delete = f'{os.path.dirname(__file__)}/data/images/'
            for filename in os.listdir(to_delete):
                os.remove(to_delete + filename)
        
    def load_graph(self, city:str = 'Vienna'):
        dirname = os.path.dirname(__file__)
        nodes = gpd.read_file(f'{dirname}/data/cities/{city}/streets.gpkg', 
                              layer='nodes')
        edges = gpd.read_file(f'{dirname}/data/cities/{city}/streets.gpkg', 
                              layer='edges')
        nodes.set_index(['osmid'], inplace=True)
        edges.set_index(['u', 'v', 'key'], inplace=True)
        nodes_m = nodes.to_crs(epsg = '32633') # metres
        edges_m = edges.to_crs(epsg = '32633') # metres
        G = ox.graph_from_gdfs(nodes, edges)
        return G, nodes_m, edges_m
        
    def _get_obs(self) -> np.ndarray:
        # Here we basically create the image
        return self.conflict_plot()
    
    def _get_info(self) -> dict:
        # Here you implement any additional info that you want to return 
        # fter a step, but that should not be used by the agent for decision 
        # making, so used for logging and debugging purposes
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
        
        # Randomise intruder number if necessary
        if self.intruders_random:
            self.n_intruders = self.np_random.integers(1, self.intr_max)
            self.n_ac = self.n_intruders + 1

        # Initialise all aircraft. Ownship will always be the one with index 0, 
        # the rest are intruders
        # Ensure the initial locations are spaced enough apart
        attempts = 10
        for attempt in range(attempts):
            self.ac_routes = self.generate_routes()
            if self.ac_routes is None:
                continue # Try again
            self.ac_locations = np.array([route[0] for route in self.ac_routes])
            self.ac_targets = np.array([route[-1] for route in self.ac_routes])
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
        
        raise EnvException('Environment could not be created. Please tweak its parameters.')
    
    def step(self, action) -> Any:
        # Get the commanded speed
        commanded_spd = self.default_speed + action * self.max_speed_diff
        # Get the difference between the commanded speed and the current speed
        spd_diff = self.ac_speeds[0] - commanded_spd
        # Can we reach this speed withon one time step?
        if abs(spd_diff) < self.mag_accel * self.dt:
            # Then set the speed as the commanded speed
            self.ac_speeds[0] = commanded_spd
        else:
            # Decrease (or increase) the speed by maximum acceleration
            if spd_diff < 0:
                self.ac_speeds[0] += self.mag_accel * self.dt
            else:
                self.ac_speeds[0] -= self.mag_accel * self.dt
        
        # Update the positions of all aircraft
        self.update_pos()
        # Get distance of the ownship to the target
        own_dist2goal = self.dist2target(0)
        # Get the distances of the ownship to other aircraft
        own_dist2others = self.dist2others(0)[1:] # skip distance to self, it's 0
        # Agent is successful if the target is reached
        success = own_dist2goal < self.target_tolerance
        # An intrusion occurs when any distance to others is smaller than the protection zone
        intrusion = bool(np.any(own_dist2others < self.rpz*2))
        self.intrusion_time_steps += 1 if intrusion else 0
        # We terminate the episode if the ownship is successful or too many time steps have passed
        terminated = success or self.step_no > self.max_steps or intrusion
        
        # We reward success and penalise intrusions
        finish_reward = 1 if success else 0 # reward for finishing
        intrusion_reward = -0.2 # Penalise intrusions
        #time_reward = -0.0001 # Penalise going very slow or standing still
        speed_reward = abs(self.ac_speeds[0]-self.default_speed) * -0.001
        reward = (finish_reward if not intrusion else 
                            finish_reward + intrusion_reward) + speed_reward
        
        # Get needed info
        observation = self._get_obs()
        info = self._get_info()
        
        self.step_no += 1
        
        return observation, reward, terminated, False, info
    
    def update_pos(self) -> None:
        # We remove intruders from the simulation area if they have reached their target
        intruders_reached = self.dist2targets() < self.target_tolerance
        # Always ignore the ownship in this part
        intruders_reached[0] = False
        for acidx in np.argwhere(intruders_reached).flatten():
            # Just place intruders that finished their flight very far away
            self.ac_locations[acidx] = [-100000,-100000]
            # And also set far away targets for em
            self.ac_targets[acidx] = [-110000,-110000]
            self.ac_current_wp[acidx] = [-110000,-110000]
            self.ac_routes[acidx][-1] = [-110000,-110000]
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
        # Check if intruders have reached their destination. If yes, remove them
        # from the picture.
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
        own_color = (0,1,0)
        # Encode its speed in the blue channel
        own_spd_color = (0,1, self.ac_speeds[0]/self.max_speed) 
        # Get route
        route = self.ac_routes[0][self.ac_wpidx[0]:]
        # Insert current location
        route.insert(0, self.ac_locations[0])
        # np arr
        route_arr = np.array(route)
        ax.plot(route_arr[:,0], route_arr[:,1],
                 color = own_color,
                 linewidth=2) # Trajectory
        
        # Plot the target
        ax.scatter(self.ac_targets[0][0],
                   self.ac_targets[0][1],
                   marker = 'x',
                   color = own_color,
                   s = 300,
                   linewidths=5)
        
        ax.scatter(self.ac_locations[0][0],
                    self.ac_locations[0][1], 
                    marker='o', 
                    color = own_spd_color,
                    s = 600,
                    linewidths=3) # Location

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
        nice_rgb = rgb_array.copy()
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
            dirname = os.path.dirname(__file__)
            fig_debug, ax = plt.subplots()
            ax.imshow(nice_rgb)
            fig_debug.savefig(f'{dirname}/data/images/{self.step_no}.png')
            fig_debug.clear()
            plt.close()
        return plot_array
        
    def close(self) -> None:
        pass
            
    def get_subgraphs_nodes(self) -> List:
        """Provides a random subgraph of the big graph in function of the city limits and playground size.

        Returns:
            Any: The subgraph
        """
        attempts = 5
        for attempt in range(attempts):
            # First, pick a random node in the big graph
            centre_node = self.nodes_m.sample()['geometry'].values[0]
            # Create a small and large square around this point
            small_square = centre_node.buffer(self.playground_size/2, cap_style = 'square')
            big_square = centre_node.buffer(self.playground_size/1.5, cap_style = 'square')
            # Get the nodes that intersect these polygons
            small_int_nodes = self.nodes_m[self.nodes_m.within(small_square)]
            big_int_nodes = self.nodes_m[self.nodes_m.within(big_square)]
            if len(big_int_nodes) > self.n_ac: # We need to spawn everyone
                return small_int_nodes.index.values.tolist(), \
                        big_int_nodes.index.values.tolist(), np.array([centre_node.x, centre_node.y])
        # return the graph anyway
        return None, None, None
        
    def generate_routes(self) -> List:
        """Generates aircraft routes based on a section of a graph.

        Returns:
            List: List of routes.
        """
        attempts = 10
        for attempt in range(attempts):
            ac_routes = []
            # We want to create two subgraphs. A smaller one for the ownship, and a larger one
            # for intruders.
            own_graph_nodes, intr_graph_nodes, centre_node = self.get_subgraphs_nodes()
            
            if own_graph_nodes is None:
                # Bad graph
                continue
            
            # We want to use the centre node to shift the coordinates to smaller
            # coords within playground bounds.
            min_x, min_y = centre_node - self.playground_size/2
            # Make graphs out of the nodes
            G_own = self.G.subgraph(own_graph_nodes)
            G_intr = self.G.subgraph(intr_graph_nodes)
            
            # Create the ownship route
            own_route, own_orig, own_dest = self.generate_route_in_graph(G_own, 
                                                        own_graph_nodes, own_graph_nodes,
                                                        min_x, min_y)
            
            if own_route is None:
                # Bad subgraphs, get others
                continue
            
            ac_routes.append(own_route.copy())
            
            # Remove the ownship origin node from the list of nodes that can be used as origins
            intr_graph_nodes.pop(intr_graph_nodes.index(own_orig))
            
            # Create the intruder routes
            for i in range(self.n_intruders):
                attempts = 5
                for attempt in range(attempts):
                    # Create the route
                    intr_route, intr_orig, intr_dest = self.generate_route_in_graph(G_intr, 
                                                        intr_graph_nodes, intr_graph_nodes,
                                                            min_x, min_y)
                    
                    if intr_route is None:
                        # Too many intruders maybe? try again
                        continue
                    
                    # Check if node is too close to ownship start point
                    dist = np.linalg.norm(np.array(ac_routes[0][0]) - np.array(intr_route[0]))
                    
                    if dist < self.rpz * 2:
                        # Too close to ownship, try again
                        continue
                    
                    ac_routes.append(intr_route.copy())
                    # Remove this origin
                    intr_graph_nodes.pop(intr_graph_nodes.index(intr_orig))
                    break
            
            if len(ac_routes) == 1:
                # We only have an ownship, try again
                continue
            
            # Update the number of intruders if they all failed to be created
            self.n_intruders = len(ac_routes) - 1
            self.n_ac = len(ac_routes)
            # return the routes
            return ac_routes
        
        # We failed to generate routes, reset
        return None
    
    def generate_route_in_graph(self, G, orig_nodes:List, dest_nodes:List,
                                min_x:float, min_y:float) -> Any:
        """Generates a route in a graph with the origin node and destination
        nodes chosen randomly from orig_nodes and dest_nodes until the global
        minimum travel dist requirement is satisfied.
        Route is also shifted to fit the playground coords using min_x, min_y.
        """
        # Try to find an origin/destination pair for the ownship
        attempts = 10
        for attempt in range(attempts):
            # Get a random origin and destination node
            org_node = self.np_random.choice(orig_nodes)
            dest_node = self.np_random.choice(dest_nodes)
            # See if there is a path between these two nodes and if the length is good
            try:
                dist = nx.shortest_path_length(G, org_node, dest_node, weight = 'length')
            except:
                # No path between them, try with other nodes
                continue
            
            if dist > self.min_travel_dist:
                # Get the actual route now
                route = nx.shortest_path(G, org_node, dest_node)
                # Make the geometry
                route_geom = linemerge([self.edges_m.loc[(u, v, 0), 'geometry'] for u, v in zip(route[:-1], route[1:])])
                # Return a nice list of waypoints
                route_wp = list(zip(route_geom.xy[0]-min_x, route_geom.xy[1]-min_y))
                return route_wp, org_node, dest_node
            
        # No route found, graph is bad
        return None, None, None
    
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
    
class EnvException(Exception):
    def __init__(self, msg):
        self.msg = msg