from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
from Game import Game
import numpy as np
import cv2
import time
from ObservationType import ObservationType, LongViewObservation
import Maps

class HideAndSeekEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface, for the game Hide and Seek.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, fps=30, map_name=Maps.DEFAULT_MAP,
                 observation_type:ObservationType=None) -> None:
        """
        Initializes the environment.
        
        Parameters
        ----------
        render_mode : str, optional, "human" or "rgb_array"
            The render mode, by default None
        fps : int, optional
            The render speed, by default 30, only used if render_mode is "human"
        map_name : str, optional
            The map to use, by default "random"
        observation_type : ObservationType, optional
            The observation type to use. If None, LongViewObservation(5) is used.

        """
        super(HideAndSeekEnv, self).__init__()
        
        self.game = Game(map_name=map_name)
        self.fps = fps

        # if no observation type is given, we use LongView as the default one
        if observation_type is None:
            observation_type = LongViewObservation(5)

        self.action_space = spaces.Discrete(4)
        self.observation_space = self._create_observation_space(observation_type.shape)
        self.observation_type = observation_type

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.info = {}
        self.steps = 0 # steps in the current episode, truncated after 300 steps
        self.maximum_steps = 300


    def _create_observation_space(self, shape) -> spaces.Box:
        """
        Creates the observation space. See ObservationType.py for observation spaces
        implementation.
        See README.md for more details about the different observation spaces.

        Parameters
        ----------
        shape : Tuple[int]
            the shape of the observation space

        Returns
        -------
        spaces.Box
            the observation space
        """

        return spaces.Box(
            low=0,
            high=max(self.game.GRID_W, self.game.GRID_H)-1,
            shape=shape,
            dtype=int
        )

    def _get_observation(self) -> np.ndarray:
        """
        Returns the current observation of the environment, based on the observation
        type.
        """
        return self.observation_type.get_observation(self.game)

    
    def _get_info(self):
        return {
            "distance": self.game.agent.pos.manhattan_distance(self.game.player.pos),
        }

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Performs the given action in the environment and returns the next observation,
        reward, and termination signal.

        Parameters
        ----------
        action : int
            the action to perform (see Game class for details)
        
        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict]
            the next observation, the reward, the termination signal,
            if episode is truncated and a dictionary of info.
        """

        self.steps+=1
        # Move the agent
        self.game.handle_action(action)
        
        # An episode is done iff the agent is hidden
        terminated = not self.game.agent.is_seen
        truncated = self.steps >= self.maximum_steps and not terminated
        reward = 50 if terminated else -1
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        

        # init the game, placing agent and player uniformly at random
        self.game.init_game_start()

        self.steps = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def render(self):
        """
        Renders the current state of the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """
        Renders the current state of the environment.
        Show the board in a window if render_mode is "human".
        """
        board = self.game.render()

        if self.render_mode == "human":
            cv2.imshow("Hide and Seek", board)

            # Takes step after fixed time, to ensure fixed framerate in human mode
            t_end = time.time() + 1/self.fps
            k = -1
            while time.time() < t_end:
                if k == -1:
                    k = cv2.waitKey(1)
                else:
                    continue

        else: # rgb_array
            return board
            # return np.transpose(
            #     board, axes=(1, 0, 2)
            # )


    def close (self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()