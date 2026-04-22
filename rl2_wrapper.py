import random
import gymnasium as gym
import numpy as np

from env import BaseGridEnv
from tasks import task_generator


class RL2MetaWrapper(gym.Env):
    """
    RL^2 wrapper for BaseGridEnv.

    - Samples one task at the start of a trial
    - Keeps that task fixed across multiple episodes
    - Augments observation with previous action, reward, done
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes_per_trial: int = 4,
        size: int = 5,
        max_steps: int = 75,
        step_penalty: float = -0.01,
        success_reward: float = 1.0,
        wrong_terminal_reward: float = 0.0,
        object_types: int = 5,
        object_colors: int = 5,
        max_depth: int = 2,
        prune_prob: float = 0.1,
        num_distractor_rules: int = 0,
        num_distractor_objects: int = 0,
    ):
        super().__init__()

        self.env = BaseGridEnv(
            size=size,
            max_steps=max_steps,
            step_penalty=step_penalty,
            success_reward=success_reward,
            wrong_terminal_reward=wrong_terminal_reward,
            object_types=object_types,
            object_colors=object_colors,
            max_depth=max_depth,
            prune_prob=prune_prob,
            num_distractor_rules=num_distractor_rules,
            num_distractor_objects=num_distractor_objects,
        )

        self.episodes_per_trial = episodes_per_trial
        self.current_episode = 0
        self.task = None

        self.python_rng = random.Random()

        self.action_space = self.env.action_space

        obs_dim = int(np.prod(self.env.observation_space.shape))

        # flat obs + prev_action + prev_reward + prev_done
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim + 3,),
            dtype=np.float32,
        )

        self.prev_action = 0.0
        self.prev_reward = 0.0
        self.prev_done = 0.0

    def _sample_task(self):
        return task_generator(
            max_depth=self.env.max_depth,
            prune_prob=self.env.prune_prob,
            num_distractor_rules=self.env.num_distractor_rules,
            num_distractor_objects=self.env.num_distractor_objects,
            object_types=list(range(self.env.object_types)),
            object_colors=list(range(self.env.object_colors)),
            rng=self.python_rng,
        )

    def _augment(self, obs):
        flat_obs = obs.astype(np.float32).flatten()
        extra = np.array(
            [self.prev_action, self.prev_reward, self.prev_done],
            dtype=np.float32,
        )
        return np.concatenate([flat_obs, extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.python_rng.seed(seed)

        self.task = self._sample_task()
        self.current_episode = 0

        obs, info = self.env.reset(seed=seed, task=self.task)

        self.prev_action = 0.0
        self.prev_reward = 0.0
        self.prev_done = 0.0

        info["trial_episode"] = self.current_episode
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        episode_done = terminated or truncated

        self.prev_action = float(action)
        self.prev_reward = float(reward)
        self.prev_done = float(episode_done)

        if episode_done:
            self.current_episode += 1

            if self.current_episode < self.episodes_per_trial:
                # same task, new episode
                obs, reset_info = self.env.reset(task=self.task)
                info.update(reset_info)

                terminated = False
                truncated = False
            # else: true end of trial

        info["trial_episode"] = self.current_episode
        return self._augment(obs), float(reward), terminated, truncated, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()