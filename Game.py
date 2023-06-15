import random

import cv2
import numpy as np

import Colors
import Maps
from Entity import Entity
from Vector2 import Vector2


class Game:
    """
    Game class. Contains the grid, the player and the agent.

    """
    def __init__(self, mode=None, map_name="statement") -> None:
        """
        Initialize the game

        Parameters
        ----------
        mode : str, optional
            "human" to play it and render it, else None (for AI training),
            by default None
        map_name : str, optional
            name of the map to load, by default "statement", the map that is in the pdf
            statement. See Maps.py for the list of available maps.
            If "random", a random map is generated. See generate_random_map() for more.
        """

        self.map_name = map_name
        self.grid = self._load_map(map_name) # contains the map
        
        
        self.GRID_W = len(self.grid[0])
        self.GRID_H = len(self.grid)


        self.SPEED = 12
        self.CELL_SIZE = 32
        self.WIDTH = self.GRID_W * self.CELL_SIZE
        self.HEIGHT = self.GRID_H * self.CELL_SIZE

        self.player = Entity(Vector2(0,0), Colors.RED)
        self.agent = Entity(Vector2(1,1), Colors.BLUE)

        self.nb_walls = 0
        self.wall_positions = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._is_wall(Vector2(x, y)):
                    self.wall_positions.append(Vector2(x, y))
                    self.nb_walls += 1
        
        # "human" to play it and render it, else None (for AI training)
        self.mode = mode

        if self.mode == "human":
            self.init_game_start()


    def _load_map(self, map_name:str) -> list:
        """
        Load the map from Maps.py.
        If map_name == "random", generate a random map instead.

        Parameters
        ----------
        map_name : str
            name of the map to load, or "random" to generate a random map

        Returns 
        -------
        list
            the map as a list of lists
        """
        if map_name != "random":
            if map_name in Maps.MAPS:
                return Maps.MAPS[map_name]
            else:
                raise ValueError(f"Map '{map_name}' does not exist. Please choose one"
                                + f" of {Maps.MAPS.keys()}")
        else:
            return self.generate_random_map(width=12, height=12)

    def generate_random_map(self, width=12, height=12, nb_walls=None) -> list:
        """
        Generate a random map
        Place randomly nb_walls walls on the grid, avoiding the border
        to not create accidently a wall that may block the player or the agent

        Parameters
        ----------
        width : int, optional
            width of the grid, by default 12
        height : int, optional
            height of the grid, by default 12
        nb_walls : int, optional
            number of walls to place, by default None,
            if None, it is random between 16 and 36 (works nice with 12x12 grid,
            but should be adjusted for other grid sizes...)
        """

        if nb_walls is None:
            nb_walls = random.randint(16, 36)
            nb_walls = 36
        
        grid = [[Maps.EMPTY for _ in range(width)] for _ in range(height)]
        

        for _ in range(nb_walls):
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)
            grid[y][x] = Maps.WALL

        return grid
        

    def init_game_start(self) -> None:
        """
        Initialize the game state (player and agent positions)
        Place player and agent at random but the player must see the agent
        """

        while True:
            self._place_entity_at_random(self.player)
            self._place_entity_at_random(self.agent)
            
             # No collision between player and agent
            if self.player.pos == self.agent.pos:
                continue

            if self.player.can_see(self.agent, self.grid):
                break
        
        self.agent.is_seen = True


    def _cell_to_pixel(self, cell:int, center=False) -> int:
        """
        Given a cell index, return the top-left pixel position of the cell.
        If center is True, return the center of the cell instead of the top-left corner.

        Parameters
        ----------
        cell : int
            cell index
        center : bool, optional
            if True, return the center of the cell instead of the top-left corner,
            by default False

        Returns
        -------
        int
            top-left pixel position of the cell
        """

        offset = self.CELL_SIZE//2 if center else 0
        return cell * self.CELL_SIZE + offset

    def _draw_grid(self, board, color=Colors.BLACK, thickness:int = 1) -> None:
        """
        Draw a grid on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        color : tuple, optional, by default Colors.BLACK, from Colors.py
            color of the grid
        thickness : int, optional, by default 1
            thickness of the grid lines
        """
        width, height, _ = board.shape 

        # draw vertical lines
        for cell_x in range(self.GRID_W):
            pixel_x = self._cell_to_pixel(cell_x) 
            cv2.line(board, (pixel_x,0), (pixel_x, height),
                     color=color,
                     thickness=thickness
            )

        # draw horizontal lines
        for cell_y in range(self.GRID_H):
            pixel_y = self._cell_to_pixel(cell_y)
            cv2.line(board, (0, pixel_y), (width, pixel_y),
                     color=color,
                     thickness=thickness
            )

    def _fill_cell(self, board, cell_x, cell_y, color) -> None:
        """
        Fill a cell with a color on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        cell_x : int
            x index of the cell
        cell_y : int
            y index of the cell
        color : tuple, from Colors.py
            color to fill the cell with
        """

        board[(cell_y*self.CELL_SIZE):((cell_y+1)*self.CELL_SIZE),
              (cell_x*self.CELL_SIZE):((cell_x+1)*self.CELL_SIZE)] = color

    def _place_entity_at_random(self, entity:Entity) -> None:
        """
        Place an entity at random on the grid, avoiding walls.

        Parameters
        ----------
        entity : Entity
            entity to place on the grid
        """

        while True:
            x = random.randint(0, self.GRID_W-1)
            y = random.randint(0, self.GRID_H-1)
            if self.grid[y][x] == Maps.EMPTY:
                entity.pos = Vector2(x, y)
                break


    def _draw_line(self, board, start_cell_x, start_cell_y,
                   target_cell_x, target_cell_y, color, thickness:int=1) -> None:
        """
        Draw a line between two cells on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        start_cell_x : int
            x index of the start cell
        start_cell_y : int
            y index of the start cell
        target_cell_x : int
            x index of the target cell
        target_cell_y : int
            y index of the target cell
        color : tuple, from Colors.py
            color of the line
        thickness : int, optional, by default 1
            thickness of the line
        """

        start_pixel_x = self._cell_to_pixel(start_cell_x, center=True)
        start_pixel_y = self._cell_to_pixel(start_cell_y, center=True)
        target_pixel_x = self._cell_to_pixel(target_cell_x, center=True)
        target_pixel_y = self._cell_to_pixel(target_cell_y, center=True)
        cv2.line(board,
                 (start_pixel_x, start_pixel_y),
                 (target_pixel_x, target_pixel_y),
                 color=color, thickness=thickness
        )

    def render(self) -> np.ndarray:
        """
        Render the game state on a display board, and returns it.

        Returns
        -------
        np.ndarray
            display board
        """
        board = np.zeros((self.WIDTH, self.HEIGHT, 3))+255 # white background

        # render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._is_wall(Vector2(x, y)):
                    self._fill_cell(board, x, y, Colors.BLACK)
                    
        # render player and agent
        self.player.draw(board)
        self.agent.draw(board)

        self._draw_grid(board)

        # Render line between player and agent
        # If the agent is seen, the line is green, otherwise it is red
        see_color = Colors.GREEN if self.agent.is_seen else Colors.RED

        self._draw_line(board, self.player.x, self.player.y,
                        self.agent.x, self.agent.y,
                        see_color, thickness=3
)
        # Display only if in human mode
        if self.mode == "human":
            cv2.imshow("Hide and Seek", board)

        return board


    def handle_inputs(self) -> bool:
        """
        Handle keyboard inputs in human mode.
        ZQSD to move the player, IJKL to move the agent.
        x, delete or escape to exit the game.
        """
        key = cv2.waitKey(int(1000/self.SPEED))

        # delete and escape keys to exit the game
        if key in [8, 27, ord('x')]:
            return True
        
        new_pos = Vector2(self.player.pos.x, self.player.pos.y)
        if key == ord('q'): # player left
            new_pos.x -= 1
        elif key == ord('d'): # player right
            new_pos.x += 1
        elif key == ord('z'): # player up
            new_pos.y -= 1
        elif key == ord('s'): # player down
            new_pos.y += 1
        if key == ord('i'): # agent up
            self.handle_action(2)
        elif key == ord('j'): # agent left
            self.handle_action(0)
        elif key == ord('k'): # agent down
            self.handle_action(3)
        elif key == ord('l'): # agent right
            self.handle_action(1)
        
        # Make the move only if the new position is valid
        if (self._is_valid_coordinates(new_pos)
            and not self._is_wall(new_pos)
            and new_pos != self.agent.pos):
            self.player.pos = new_pos

        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False

        return False
    
    def _is_valid_coordinates(self, coord:Vector2) -> bool:
        """
        Check if the coordinates are valid, i.e. inside the grid.
        
        Parameters
        ----------
        coord : Vector2
            coordinates to check

        Returns
        -------
        bool
            True if the coordinates are valid, False otherwise
        """
        return (coord.x >= 0 
                and coord.x < self.GRID_W
                and coord.y >= 0
                and coord.y < self.GRID_H)

    def _is_wall(self, coord:Vector2) -> bool:
        """
        Check if the coordinates are a wall.

        Parameters
        ----------
        coord : Vector2
            coordinates to check

        Returns
        -------
        bool
            True if the coordinates are a wall, False otherwise
        """
        return self.grid[coord.y][coord.x] == Maps.WALL


    def handle_action(self, action) -> None:
        """
        Handle the action of the agent.
        The agent cannot move through walls and cannot move outside the grid.
        The action are:
            0: move left
            1: move right
            2: move up
            3: move down
        """
        new_pos = Vector2(self.agent.pos.x, self.agent.pos.y)
        if action == 0:
            new_pos.x -= 1
        elif action == 1:
            new_pos.x += 1
        elif action == 2:
            new_pos.y -= 1
        elif action == 3:
            new_pos.y += 1
        
        if (self._is_valid_coordinates(new_pos)
            and not self._is_wall(new_pos)
            and new_pos != self.player.pos):
            self.agent.pos = new_pos
        
        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False


    def run(self) -> None:
        """
        Run the game.
        """
        print("Run")
        while True:
            self.render()

            quit = self.handle_inputs()
            if quit:
                break
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = Game(mode="human")
    game.run()
