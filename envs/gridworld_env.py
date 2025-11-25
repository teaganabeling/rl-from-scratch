import numpy as np
import matplotlib.pyplot as plt # for rendering

from gymnasium import spaces
from envs.environment import Environment
from core.config import load_config
from core.utils import set_seed

frames = []

MAPS = {
    "3x4": [
        "0000",
        "0X00",
        "0X00"
    ],

    "6x6": [
        "0000X0",
        "0000X0",
        "0XX000",
        "000000",
        "000XX0",
        "0000X0"
    ],

    "12x12": [
        "000000X0X000",
        "000XXXX00000",
        "0X00000000XX",
        "00000000XX00",
        "X00XX0000001",
        "X00000000000",
        "XX00XX00XXXX",
        "0000X000X000",
        "X000X0000X00",
        "X000X0X00000",
        "0000X00000X0",
        "00X0000XXX0X"
    ]
}

config = load_config()
set_seed(config["env"]["seed"])

class GridworldEnv(Environment):

    FREE: str = '0' # char for free cell
    WALL: str = 'X' # char for wall cell
    MOVES: dict[int, tuple] = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1)    # Right
    } 

    # Create Gridworld
    def __init__(self, obstacle_map: str | list[str]):
        self.obstacles = self.parse_obstacle_map(obstacle_map)
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.state_space = spaces.Discrete(self.nrow * self.ncol)

        self.goal_pos = (0, self.ncol - 1)  # Goal position
        self.agent_pos = (self.nrow - 1, 0) # Start position

        self.reward_map = self.create_reward_map()
    
        
    def create_reward_map(self):
        free_reward = config['gridworld']['cell_rewards']['free']
        wall_reward = config['gridworld']['cell_rewards']['wall']
        goal_reward = config['gridworld']['cell_rewards']['goal']


        reward_map = np.full((self.nrow, self.ncol), free_reward, dtype=float) # default: -1 per step (living cost)
        reward_map[self.obstacles == self.WALL] = wall_reward # Walls have no reward
        gx, gy = self.goal_pos
        reward_map[gx, gy] = goal_reward
        return reward_map
    
    def save_reward_map_as_csv(self, filename="reward_map.csv"):
        import pandas as pd
        df = pd.DataFrame(self.reward_map)
        df.to_csv(filename, index=False, header=False)
    
    def parse_obstacle_map(self, obstacle_map):
        if isinstance(obstacle_map, str):
            if obstacle_map not in MAPS:
                raise ValueError(f"Unknown map key: {obstacle_map}")
            obstacle_map = MAPS[obstacle_map]

        # Convert each row string into a list of integers
        grid = np.array([[cell for cell in row] for row in obstacle_map])
        return grid
    
    def reset(self):
        self.agent_pos = (self.nrow - 1, 0)
        return self.agent_pos

    def step(self, action):
        move = self.MOVES[action]
        next_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # Check bounds and obstacles
        if (0 <= next_pos[0] < self.nrow and
            0 <= next_pos[1] < self.ncol and
            self.obstacles[next_pos] == self.FREE):
            self.agent_pos = next_pos

        reward = self.reward_map[self.agent_pos] # Get reward for current position

        done = self.agent_pos == self.goal_pos # Check terminal condition

        return self.agent_pos, reward, done, {}


    def simple_render(self):
        grid = np.copy(self.obstacles)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        
        grid[ax, ay] = 'A'  # Agent
        grid[gx, gy] = 'G' # Goal
        print(grid)

    def render(self, mode="human"):
        if not hasattr(self, "fig"): 
            self.fig, self.ax = plt.subplots() # create a figure and axis
            
            self.img = self.ax.imshow(np.ones((self.nrow, self.ncol, 3)), origin="upper", interpolation="nearest") # initialize with a white image
            
            self.ax.set_xlim(-0.5, self.ncol - 0.5)
            self.ax.set_ylim(self.nrow - 0.5, -0.5)  # top-left origin
            
            self.ax.set_xticks(np.arange(self.ncol))
            self.ax.set_yticks(np.arange(self.nrow))
            self.ax.set_xticklabels([str(i) for i in range(self.ncol)])
            self.ax.set_yticklabels([str(i) for i in range(self.nrow)])
            
            self.ax.set_xticks(np.arange(-0.5, self.ncol, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.nrow, 1), minor=True)
            self.ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
            
            self.ax.tick_params(which="both", length=0)
            self.ax.set_aspect("equal", adjustable="box")

            plt.ion() # interactive mode
            plt.show() # show the plot

        img = np.ones((self.nrow, self.ncol, 3))
        colors = {
            'free': (0.95, 0.95, 0.95),
            'wall': (0.2, 0.2, 0.2),
            'agent': (0.3, 0.6, 0.9),
            'goal': (0.4, 0.8, 0.4)
        }

        for row in range(self.nrow):
            for column in range(self.ncol):
                if (row, column) == self.agent_pos:
                    img[row, column] = colors['agent']
                elif (row, column) == self.goal_pos:
                    img[row, column] = colors['goal']
                elif self.obstacles[row, column] == self.WALL:
                    img[row, column] = colors['wall']
                else:
                    img[row, column] = colors['free']

        self.img.set_data(img) # update the image data
        self.fig.canvas.draw() # redraw the canvas
        self.fig.canvas.flush_events() # flush the GUI events for real-time updates

