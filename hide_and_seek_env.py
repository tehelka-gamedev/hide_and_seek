import gymnasium as gym
from gymnasium import spaces

from Game import Game

################# Hide and Seek GAME

import numpy as np
import cv2
import random
import time
from collections import deque

from Vector2 import Vector2

####################


class HideAndSeekEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, fps=30):
        super(HideAndSeekEnv, self).__init__()
        
        self.game = Game()
        self.fps = fps

        nb_walls = self.game.nb_walls

        self.action_space = spaces.Discrete(4)

        # my version
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.game.GRID_W, self.game.GRID_H)-1,
            shape=(5,),
            #shape=(5 + 2*nb_walls,),
            dtype=int
        )

        # my version v2
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.game.GRID_W, self.game.GRID_H)-1,
            shape=(5+8,),
            #shape=(5 + 2*nb_walls,),
            dtype=int
        )

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.observation = None
        self.reward = 0
        self.done = False
        self.info = {}




   

    def _get_observation(self):
        """
        Returns the current observation of the environment.
        The obersevation is a tuple of size 5 of the form:
        (player_x, player_y, agent_x, agent_y, agent_is_seen)

        + 8 booleans (0 or 1) if the surrounding cells are walls or not
        """
        obs = [
            self.game.player.pos.x, self.game.player.pos.y,
            self.game.agent.pos.x, self.game.agent.pos.y,
            int(self.game.agent.is_seen),
        ]
        
        # add surrounding cells info to the observation
        # 1 if wall, 0 if not
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                coord = Vector2(self.game.agent.pos.x + i, self.game.agent.pos.y + j)
                if self.game._is_valid_coordinates(coord) and self.game._is_wall(coord):
                    obs.append(1)
                else:
                    obs.append(0)
        
        # for wall_pos in self.game.wall_positions:
        #     obs += [wall_pos.x, wall_pos.y]

        return np.array(obs)

    def _get_observation_v1(self):
        """
        Returns the current observation of the environment.
        The obersevation is a tuple of size 5+2*nb_walls of the form:
        (player_x, player_y, agent_x, agent_y, agent_is_seen, wall_1_x, wall_1_y, wall_2_x, wall_2_y, ...)
        """
        obs = [
            self.game.player.pos.x, self.game.player.pos.y,
            self.game.agent.pos.x, self.game.agent.pos.y,
            int(self.game.agent.is_seen),
        ]
        
        # for wall_pos in self.game.wall_positions:
        #     obs += [wall_pos.x, wall_pos.y]

        return np.array(obs)
    
    def _get_info(self):
        return {
            "distance": self.game.agent.pos.manhattan_distance(self.game.player.pos),
        }

    def step(self, action):
        # self.prev_actions.append(action)


        # the agent moves
        self.game.handle_action(action)
        
        # An episode is done iff the agent is hidden
        terminated = not self.game.agent.is_seen
        reward = 10 if terminated else -1
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        # self.game.render()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        

        # init the game, placing agent and player uniformly at random
        self.game.init_game()

        self.done = False
        self.reward = 0


        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
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