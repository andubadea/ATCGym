import numpy as np
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
import os
import imageio.v2 as imageio
import re
import osmnx as ox
import networkx as nx
from shapely.ops import linemerge
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple, Dict, Any, List
from scipy.spatial.distance import pdist, squareform
matplotlib.rcParams['interactive'] = False
matplotlib.use('Agg')

class ConflictMultiArtEnv(gym.Env):
    """This environment creates conflicts with drones that need resolving.
    """
    
    metadata = {"render_modes": ["images"], 
                "render_fps": 4, 
                "image_mode": ["rgb", "rel_rgb", "rel_gry"]}
    
    def __init__(self, render_mode=None, n_intruders = 3, 
                                image_mode = 'rgb', image_pixel_size = 128):
        # Will want to eventually make these env properties
        self.ac_max = 3
        self.n_aircraft = n_intruders
        self.playground_size = 200 # metres, also square
        self.min_travel_dist = 100 #metres, minimum travel distance
        self.rpz = 16 #metres, protection zone radius
        self.mag_accel = 3.5 # m/s, constant acceleration magnitude
        self.max_speed = 18 #m/s, maximum speed
        self.default_speed = 9 #m/s, starting speed for ownship
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
        self.reached_goal = None
        
        assert image_mode in self.metadata["image_mode"]
        self.image_mode = image_mode
        
        # Assign one aircraft per image layer
        self.observation_space = spaces.Box(low = 0, high = 255, 
                                    shape=(self.image_pixel_size,
                                            self.image_pixel_size, 4), #rgba
                                    dtype=np.uint8)
        
        # 0 = default speed, -1 = min speed, 1 = max speed
        self.action_space = spaces.Box(low = -1, high = 1, 
                                       shape=(self.n_aircraft,1))
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.eval_no = 1
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
        
    def _get_obs(self, is_init = False) -> np.ndarray:
        # Here we basically create the image
        return self.conflict_plot(is_init)
    
    def _get_info(self) -> dict:
        # Here you implement any additional info that you want to return 
        # fter a step, but that should not be used by the agent for decision 
        # making, so used for logging and debugging purposes
        # Initialise info dict
        info_dict = {}
        # Intruder infos
        for acidx in range(self.n_aircraft):
            info_dict[f"{acidx}_location"] = self.ac_locations[acidx]
            info_dict[f"{acidx}_target"] = self.ac_targets[acidx]
            info_dict[f"{acidx}_dist"] = self.dist2target(acidx)
        return info_dict
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        # This generates a new episode after one has been completed
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.step_no = 0
            
        if self.render_mode == "images":
            # Get a list of all the images in the debug folder
            png_folder = f'{os.path.dirname(__file__)}/data/images/'
            png_list = self.natural_sort([img for img in os.listdir(png_folder) 
                                                if '.png' in img])
            if len(png_list) > 1:
                # Create a gif
                images = []
                for img in png_list:
                    images.append(imageio.imread(png_folder + img))
                imageio.mimsave(f'output/Eval_{self.eval_no}.gif', 
                                images)
                self.eval_no += 1
            
            # Clean up
            for filename in os.listdir(png_folder):
                os.remove(png_folder + filename)

        # Initialise all aircraft. Ensure the initial locations are spaced 
        # enough apart
        attempts = 10
        for attempt in range(attempts):
            self.ac_routes = self.generate_routes()
            if self.ac_routes is None:
                continue # Try again
            self.ac_locations = np.array(
                [route[0] for route in self.ac_routes])
            self.ac_targets = np.array(
                [route[-1] for route in self.ac_routes])
            self.ac_wpidx = np.ones(
                self.n_aircraft, dtype=int) # waypoint index
            self.ac_current_wp = np.array(
                [route[1] for route in self.ac_routes]) # target is first wp
            # Create the speed array
            self.ac_speeds = np.ones(self.n_aircraft) * self.default_speed
            # Initialised the reached array
            self.reached_goal = [False]*self.n_aircraft

            observation = self._get_obs(is_init=True)
            info = self._get_info()

            return observation, info
        
        raise EnvException('Environment could not be created. Please tweak its parameters.')
    
    def step(self, action) -> Any:
        # Get the commanded speed
        commanded_spd = self.default_speed + action.flatten() * \
                                                        self.max_speed_diff
        # Get the difference between the commanded speed and the current speed
        spd_diff = self.ac_speeds - commanded_spd
        # Can we reach this speed withon one time step?
        self.ac_speeds = np.where(spd_diff < self.mag_accel*self.dt,
                    commanded_spd,
                    self.ac_speeds-np.sign(spd_diff)*self.mag_accel*self.dt)
        
        # Update the positions of all aircraft
        ac_reached = self.update_pos()
        # An intrusion occurs when any distance to others is smaller than 
        # the protection zone
        dist_vec = pdist(self.ac_locations)
        intrusion_num = sum(dist_vec < self.rpz*2)
        intrusion = intrusion_num > 0
        # We terminate the episode if all aircraft reached their goals 
        # succesfully
        success = np.all(self.reached_goal)
        terminated = success or self.step_no > self.max_steps or intrusion
        # We reward success and penalise intrusions
        finish_reward = sum(ac_reached) # reward for finishing
        intrusion_reward = -0.1 * intrusion_num # Penalise intrusions
        #time_reward = -0.0001 # Penalise going very slow or standing still
        speed_reward = sum(np.abs(self.ac_speeds-self.default_speed) * -0.0001)
        reward = finish_reward + intrusion_reward + speed_reward
        
        # Get needed info
        observation = self._get_obs()
        info = self._get_info()
        
        self.step_no += 1
        
        return observation, reward, terminated, False, info
    
    def update_pos(self) -> None:
        # We remove aircraft from the simulation area if they have reached 
        # their target
        ac_reached = np.argwhere(self.dist2targets() < self.target_tolerance
                                 ).flatten()
        
        for acidx in ac_reached:
            # Just place intruders that finished their flight very far away
            f = acidx + 1
            self.ac_locations[acidx] = [-10000*f,-10000*f]
            # And also set far away targets for em
            self.ac_targets[acidx] = [-11000*f,-11000*f]
            self.ac_current_wp[acidx] = [-11000*f,-11000*f]
            self.ac_routes[acidx][-1] = [-11000*f,-11000*f]
            # Mark this one as complete
            self.reached_goal[acidx] = True
            
        # Get distances to current waypoints
        dist_wps = self.dist2curwps()
        # Aircraft that reached their wp switch to the next one
        for acidx in np.argwhere(dist_wps < self.target_tolerance).flatten():
            self.ac_wpidx[acidx] += 1
            # Only update waypoint if it's not the last waypoint
            if self.ac_wpidx[acidx] < len(self.ac_routes[acidx]):
                self.ac_current_wp[acidx] = self.ac_routes[acidx][
                                                        self.ac_wpidx[acidx]]
        
        # Get the direction vectors for all aircraft directions
        dir_v = self.ac_current_wp - self.ac_locations
        # Get unit vector
        unit_dir_v = np.transpose(dir_v.T / np.linalg.norm(dir_v, axis = 1))
        # Now update position
        self.ac_locations += np.transpose(
                            (self.ac_speeds * self.dt) * unit_dir_v.T)
        # Check if intruders have reached their destination. If yes, remove them
        # from the picture.
        return ac_reached
        
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
        return np.linalg.norm(self.ac_locations[acidx] - self.ac_locations, 
                              axis=1)
    
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
        
    
    def conflict_plot(self, is_init = False) -> np.ndarray:
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
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, 
                            hspace=0)
        
        # Encode the velocity of the aircraft within
        ac_spd_color = self.ac_speeds/self.max_speed
    
        # Plot the aircraft within the rgb channels
        for acidx in range(self.n_aircraft):
            color = [0,0,0,1] # Initialise the color array
            color[acidx] = 1
            spd_color = color.copy()
            spd_color[-1] = ac_spd_color[acidx] # Encode speed in alpha
            ax.scatter(self.ac_locations[acidx][0],
                        self.ac_locations[acidx][1], 
                        marker='o', 
                        color = spd_color,
                        s = self.rpz**2*55) # Location
            
            # Get route
            route = self.ac_routes[acidx][self.ac_wpidx[acidx]:]
            # Insert current location
            route.insert(0, self.ac_locations[acidx])
            # np arr
            route_arr = np.array(route)
            ax.plot(route_arr[:,0], route_arr[:,1],
                    color = color,
                    linewidth=2) # Trajectory
            
            # Plot the target
            ax.scatter(self.ac_targets[acidx][0],
                    self.ac_targets[acidx][1],
                    marker = 'x',
                    color = color,
                    s = 300,
                    linewidths=10)

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
        rgb_array[self.image_pixel_size-1,:] = np.zeros((self.image_pixel_size, 
                                                         4)) + 255
        rgb_array[:, 0] = np.zeros((self.image_pixel_size, 4)) + 255
        rgb_array[:, self.image_pixel_size-1] = np.zeros((self.image_pixel_size, 
                                                          4)) + 255
        nice_rgb = rgb_array[:,:,:3]
        # Invert all the colors, 255 becomes 0
        rgb_array = np.abs(rgb_array-255)
        # Clear memory
        fig.clear()
        plt.close()
        
        if self.render_mode == "images" and not is_init:
            dirname = os.path.dirname(__file__)
            fig_debug, ax = plt.subplots()
            ax.imshow(nice_rgb)
            fig_debug.savefig(f'{dirname}/data/images/{self.step_no}.png')
            fig_debug.clear()
            plt.close()
        return rgb_array
        
    def close(self) -> None:
        pass
            
    def get_subgraphs_nodes(self) -> List:
        """Provides a random subgraph of the big graph in function of the city 
        limits and playground size.

        Returns:
            Any: The subgraph
        """
        attempts = 5
        for attempt in range(attempts):
            # First, pick a random node in the big graph
            centre_node = self.nodes_m.sample()['geometry'].values[0]
            # Create a small and large square around this point
            small_square = centre_node.buffer(self.playground_size/2, 
                                              cap_style = 'square')
            big_square = centre_node.buffer(self.playground_size/1.5, 
                                            cap_style = 'square')
            # Get the nodes that intersect these polygons
            small_int_nodes = self.nodes_m[self.nodes_m.within(small_square)]
            big_int_nodes = self.nodes_m[self.nodes_m.within(big_square)]
            if len(big_int_nodes) > self.n_aircraft:
                return small_int_nodes.index.values.tolist(), \
                        big_int_nodes.index.values.tolist(), \
                        np.array([centre_node.x, centre_node.y])
        # return nothing
        return None, None, None
        
    def generate_routes(self) -> List:
        """Generates aircraft routes based on a section of a graph.

        Returns:
            List: List of routes.
        """
        attempts = 50
        for attempt in range(attempts):
            ac_routes = []
            # Get subgraph
            all_nodes, centre_nodes, centre_coords = self.get_subgraphs_nodes()
            origin_nodes = all_nodes.copy()
            origin_coords = []
            min_x, min_y = centre_coords - self.playground_size/2
            subG = self.G.subgraph(all_nodes)
            for acidx in range(self.n_aircraft):
                # Create an aircraft route
                ac_route, ac_orig, ac_dest = self.generate_route_in_graph(
                                                subG, origin_nodes, all_nodes,
                                                min_x, min_y)
            
                if ac_route is None:
                    # Bad subgraph, try again
                    break
                elif len(origin_coords) > 1:
                    # Check the distance between the origins
                    tmp_origins = origin_coords.copy()
                    tmp_origins.append(ac_route[0])
                    if np.any(pdist(tmp_origins) < self.rpz*4):
                        # Try again
                        break
                    
                # Made it past the checks, append.
                ac_routes.append(ac_route.copy())
            
                # Remove this origin node from the list of nodes that can be 
                # used as origins
                origin_nodes.pop(origin_nodes.index(ac_orig))
            # Do we have enough routes?
            if len(ac_routes) == self.n_aircraft:
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
            # See if there is a path between these two nodes and if the length 
            # is good
            try:
                dist = nx.shortest_path_length(G, org_node, dest_node, 
                                               weight = 'length')
            except:
                # No path between them, try with other nodes
                continue
            
            if dist > self.min_travel_dist:
                # Get the actual route now
                route = nx.shortest_path(G, org_node, dest_node)
                # Make the geometry
                route_geom = linemerge([self.edges_m.loc[(u, v, 0), 
                        'geometry'] for u, v in zip(route[:-1], route[1:])])
                # Return a nice list of waypoints
                route_wp = list(zip(route_geom.xy[0]-min_x, 
                                    route_geom.xy[1]-min_y))
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
    
    @staticmethod
    def natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in 
                                                re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)  
    
class EnvException(Exception):
    def __init__(self, msg):
        self.msg = msg