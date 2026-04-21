import random
from typing import Optional
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

'''NOTES:
guard against sampling more unique (type color) objects than object space allows'''
from tasks import task_generator

@dataclass
class GridObject:
        type_id: int
        color_id:int
        pos: np.ndarray

class BaseGridEnv(gym.Env):

    
    # note that the paper uses max_steps = 3 * width * height but our env is simpler
    def __init__(self, size: int = 5, max_steps: int = 75, step_penalty: float = -0.01,
                 success_reward: float = 1.0, wrong_terminal_reward: float = 0.0,
                 object_types : int = 3, object_colors : int = 4, max_depth : int = 2, 
                 prune_prob : float = 0.1, num_distractor_rules : int = 2, num_distractor_objects : int = 2):
        
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
        self.max_depth = max_depth
        self.prune_prob = prune_prob
        self.num_distractor_rules = num_distractor_rules
        self.num_distractor_objects = num_distractor_objects


        max_unique = self.object_types * self.object_colors

        # rough upper bound on objects needed (binary tree worst case)
        max_required = 2 * (2 ** (self.max_depth + 1) - 1)

        if max_unique < max_required:
            raise ValueError(
                f"Object space too small: need ~{max_required}, have {max_unique}"
            )


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
    
    def sample_empty_cell(self, occupied: set[tuple[int, int]]) -> np.ndarray:
        while True:
            pos = (
                int(self.grid_position_rng.integers(0, self.size)),
                int(self.grid_position_rng.integers(0, self.size)),
            )
            if pos not in occupied:
                return np.array(pos, dtype=np.int32)


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.step_count = 0
        self.task_progress = 0   
        self.objects : list[GridObject] = []
        self.agent_pos = None

        self.grid_position_rng = np.random.default_rng(seed) #for grid positions
        self.rng = random.Random(seed) #for task generator

        self.task = task_generator(
                max_depth=self.max_depth,
                prune_prob=self.prune_prob,
                num_distractor_rules=self.num_distractor_rules,
                num_distractor_objects=self.num_distractor_objects,
                object_types=list(range(self.object_types)),
                object_colors=list(range(self.object_colors)),
                rng=self.rng,
                )

        occupied: set[tuple[int, int]] = set()

        self.agent_pos = self.sample_empty_cell(occupied)
        occupied.add((int(self.agent_pos[0]), int(self.agent_pos[1])))


        for leaf_object in self.task["leaf_task_nodes"]:
            
            pos = self.sample_empty_cell(occupied)
            occupied.add((int(pos[0]), int(pos[1])))
            self.objects.append(
                GridObject(
                type_id= leaf_object["type"],
                color_id= leaf_object["color"],
                pos=pos,
                )
            )


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

    def objects_match(self, obj: GridObject, spec: dict) -> bool:
        return obj.type_id == spec["type"] and obj.color_id == spec["color"]

    def are_adjacent(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        return abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1])) == 1
    
    
    def apply_rules(self) -> bool:
        for node in self.task["all_task_nodes"]:
            if node["kind"] != "rule":
                continue

            match1 = None
            match2 = None

            for obj in self.objects:
                if match1 is None and self.objects_match(obj, node["input_object_1"]):
                    match1 = obj
                elif match2 is None and self.objects_match(obj, node["input_object_2"]):
                    match2 = obj

            if match1 is not None and match2 is not None:
                if self.are_adjacent(match1.pos, match2.pos):
                    new_pos = match1.pos.copy()
                    self.objects.remove(match1)
                    self.objects.remove(match2)
                    self.objects.append(
                        GridObject(
                            type_id=node["output_object"]["type"],
                            color_id=node["output_object"]["color"],
                            pos=new_pos,
                        )
                    )
                    return True

        return False
    
    def check_goal(self) -> bool:
        goal = self.task["all_task_nodes"][0]

        goal_obj1 = None
        goal_obj2 = None

        for obj in self.objects:
            if goal_obj1 is None and self.objects_match(obj, goal["object_1"]):
                goal_obj1 = obj
            elif goal_obj2 is None and self.objects_match(obj, goal["object_2"]):
                goal_obj2 = obj

        return (
            goal_obj1 is not None
            and goal_obj2 is not None
            and self.are_adjacent(goal_obj1.pos, goal_obj2.pos)
        )
    
    def step(self, action):
        self.step_count += 1
        reward = self.step_penalty
        terminated = False
        truncated = False

        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])

        if action == 0:      # up
            nx, ny = x - 1, y
        elif action == 1:    # down
            nx, ny = x + 1, y
        elif action == 2:    # left
            nx, ny = x, y - 1
        elif action == 3:    # right
            nx, ny = x, y + 1
        else:
            raise ValueError("Invalid action")

        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.agent_pos = np.array([nx, ny], dtype=np.int32)

        self.apply_rules()

        if self.check_goal():
            reward = self.success_reward
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        return self.get_obs(), reward, terminated, truncated, {}

            



            
