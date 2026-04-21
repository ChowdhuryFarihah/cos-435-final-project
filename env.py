import random
from typing import Optional
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


from tasks import sample_task

@dataclass
class GridObject:
        type_id: int
        color_id:int
        pos: np.ndarray

class BaseGridEnv(gym.Env):

    
    # note that the paper uses max_steps = 3 * width * height but our env is simpler
    def __init__(self, size: int = 5, max_steps: int = 75, step_penalty: float = -0.01,
                 success_reward: float = 1.0, wrong_terminal_reward: float = 0.0,
                 object_types : int = 3, object_colors : int = 4):
        
        if object_types <= 0:
            raise ValueError("object_types must be positive")
        if object_colors <= 0:
            raise ValueError("object_colors must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.wrong_terminal_reward = wrong_terminal_reward
        self.object_types = object_types
        self.object_colors = object_colors

        self.objects : list[GridObject] = []
        self.agent_pos = None

        self.step_count = 0
        self.task = None

        # we have an explicit task tracking where as in paper it was implicit 
        self.task_progress = 0

        # 13 channels: 1 for agent location + 3*4 channels for different combinations
        # of objects
        self.num_channels = 1 + self.object_colors*self.object_types


        # 4 actions: up, down, left, right. Agent collects an object only when it steps onto 
        # the location of the object, at which point the position of the object and the agent 
        # until the agent is near 
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.size, self.size, self.num_channels),
            dtype=np.float32,
            
        )

    def mapping_object_to_channel(self, type_id:int, color_id:int) -> int:
         return 1 + type_id*self.object_colors + color_id
    
    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> np.ndarray:
        while True:
            pos = (
                int(self.rng.integers(0, self.size)),
                int(self.rng.integers(0, self.size)),
            )
            if pos not in occupied:
                return np.array(pos, dtype=np.int32)


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.step_count = 0
        self.task_progress = 0   
        self.objects : list[GridObject] = []
        self.agent_pos = None

        self.rng = np.random.default_rng(seed)

        self.task = {
        "type": "reach",
        "target_index": 0  # will point to first object
        }

        occupied: set[tuple[int, int]] = set()

        self.agent_pos = self._sample_empty_cell(occupied)
        occupied.add((int(self.agent_pos[0]), int(self.agent_pos[1])))

        num_objects = 3 

        for i in range(num_objects):
            type_id = int(self.rng.integers(0, self.object_types))
            color_id = int(self.rng.integers(0, self.object_colors))
            pos = self.sample_empty(occupied)
            occupied.add((int(self.agent_pos[0]), int(self.agent_pos[1])))


        obj = GridObject(type_id, color_id, pos)
        self.objects.append(obj)

        return self.get_obs(), {}

        
    def get_obs(self):
        
        obs = np.zeros((self.size, self.size, self.num_channels), dtype=np.float32)

        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        obs[agent_x, agent_y, 0] = 1.0

        for obj in self.objects:
            x, y = int(obj.pos[0]), int(obj.pos[1])
            c = self.mapping_object_to_channel(obj.type_id, obj.color_id)
            obs[x, y, c] = 1.0

        return obs



        



        
