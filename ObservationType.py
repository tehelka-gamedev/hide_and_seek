from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from Game import Game
from Vector2 import Vector2

class ObservationType(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_observation(self, game:Game) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class BasicObservation(ObservationType):
    def __init__(self) -> None:
        super().__init__()
        self.shape = (5,)

    def get_observation(self, game:Game) -> np.ndarray:
        """
        Returns the current observation of the environment.
        The obersevation is a tuple of size 5 the form:
        (player_x, player_y, agent_x, agent_y, agent_is_seen)
        
        Returns
        -------
        np.ndarray
            the observation
        """
        obs = [
            game.player.pos.x, game.player.pos.y,
            game.agent.pos.x, game.agent.pos.y,
            int(game.agent.is_seen),
        ]

        return np.array(obs)

class ImmediateSuroundingsObservation(ObservationType):
    def __init__(self) -> None:
        super().__init__()
        self.shape = (5+8,)

    def get_observation(self, game:Game) -> np.ndarray:
        """
        Returns the current observation of the environment.
        The obersevation is a tuple of size 5 of the form:
        (player_x, player_y, agent_x, agent_y, agent_is_seen)

        + 8 booleans (0 or 1) if the surrounding cells are walls or not
        """
        obs = [
            game.player.pos.x, game.player.pos.y,
            game.agent.pos.x, game.agent.pos.y,
            int(game.agent.is_seen),
        ]
        
        # add surrounding cells info to the observation
        # 1 if wall, 0 if not
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                coord = Vector2(game.agent.pos.x + i, game.agent.pos.y + j)
                if game._is_valid_coordinates(coord) and game._is_wall(coord):
                    obs.append(1)
                else:
                    obs.append(0)

        return np.array(obs)


class LongViewObservation(ObservationType):
    def __init__(self, view_size=5) -> None:
        super().__init__()
        self.shape = (5+8*view_size,)
        self.view_size = view_size

    def get_observation(self, game:Game) -> np.ndarray:
        """
        Returns the current observation of the environment.
        The obersevation is a tuple of size 5 of the form:
        (player_x, player_y, agent_x, agent_y, agent_is_seen)

        + 8*self.view_size booleans (0 or 1) walls are "nearby" or not
        For each 4 directions + 4 diagonals, cast a ray and remember which cells are
        walls

        Returns
        -------
        np.ndarray
            the observation
        """
        obs = [
            game.player.pos.x, game.player.pos.y,
            game.agent.pos.x, game.agent.pos.y,
            int(game.agent.is_seen),
        ]
        
        # cast rays in 4 directions + 4 diagonals
        # horizontal
        for dx in range(1, self.view_size+1):
            coord = Vector2(game.agent.pos.x + dx, game.agent.pos.y)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)

            coord = Vector2(game.agent.pos.x - dx, game.agent.pos.y)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)
        
        # vertical
        for dy in range(1, self.view_size+1):
            coord = Vector2(game.agent.pos.x, game.agent.pos.y + dy)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)

            coord = Vector2(game.agent.pos.x, game.agent.pos.y - dy)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)
        
        # diagonal
        for d in range(1, self.view_size+1):
            coord = Vector2(game.agent.pos.x + d, game.agent.pos.y + d)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)

            coord = Vector2(game.agent.pos.x - d, game.agent.pos.y + d)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)

            coord = Vector2(game.agent.pos.x + d, game.agent.pos.y - d)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)

            coord = Vector2(game.agent.pos.x - d, game.agent.pos.y - d)
            if game._is_valid_coordinates(coord) and game._is_wall(coord):
                obs.append(1)
            else:
                obs.append(0)


        return np.array(obs)

    def __str__(self) -> str:
        return self.__class__.__name__ + f"(view_size={self.view_size})"