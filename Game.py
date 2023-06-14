import numpy as np
import cv2
import random
import time
from Vector2 import Vector2
from Entity import Entity
import Colors



    


class Game:
    def __init__(self, mode=None) -> None:
        self.grid = [
            "............",
            "..#....#....",
            "..###..###..",
            "..###..##...",
            "..###.......",
            "............",
            "............",
            "...##..####.",
            "..###..###..",
            "..###..###..",
            "....#...##..",
            "............",
        ]

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
                if self.grid[y][x] == "#":
                    self.wall_positions.append(Vector2(x, y))
                    self.nb_walls += 1
        
        # "human" to play it and render it, else None (for AI training)
        self.mode = mode


    def init_game(self) -> None:
        # place player and agent at random but the player must see the agent

        board = np.zeros((self.WIDTH, self.HEIGHT, 3))+255 # white background
        # render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y][x] == "#":
                    self._fill_cell(board, x, y, Colors.BLACK)

        while True:
            self._place_entity_at_random(self.player)
            self._place_entity_at_random(self.agent)

            if self.player.pos == self.agent.pos:
                continue

            if self.player.can_see(self.agent, self.grid):
                break
        


        self.agent.is_seen = True

    def _cell_to_pixel(self, cell:int, center=False) -> int:
        offset = self.CELL_SIZE//2 if center else 0
        return cell * self.CELL_SIZE + offset

    def _draw_grid(self, board, color=Colors.BLACK, thickness:int = 1) -> None:
        width, height, _ = board.shape 

        # draw vertical lines
        for cell_x in range(self.GRID_W):
            pixel_x = self._cell_to_pixel(cell_x) 
            cv2.line(board, (pixel_x,0), (pixel_x, height), color=color, thickness=thickness)

        # draw horizontal lines
        for cell_y in range(self.GRID_H):
            pixel_y = self._cell_to_pixel(cell_y)
            cv2.line(board, (0, pixel_y), (width, pixel_y), color=color, thickness=thickness)

    def _fill_cell(self, board, cell_x, cell_y, color) -> None:
        # @TODO: REFACTO ?
        board[(cell_y*self.CELL_SIZE):((cell_y+1)*self.CELL_SIZE), (cell_x*self.CELL_SIZE):((cell_x+1)*self.CELL_SIZE)] = color

    def _place_entity_at_random(self, entity:Entity) -> None:
        while True:
            x = random.randint(0, self.GRID_W-1)
            y = random.randint(0, self.GRID_H-1)
            if self.grid[y][x] == ".":
                entity.pos = Vector2(x, y)
                break


    def _draw_line(self, board, start_cell_x, start_cell_y, target_cell_x, target_cell_y, color, thickness:int=1) -> None:
        
        start_pixel_x, start_pixel_y = self._cell_to_pixel(start_cell_x, center=True), self._cell_to_pixel(start_cell_y, center=True)
        target_pixel_x, target_pixel_y = self._cell_to_pixel(target_cell_x, center=True), self._cell_to_pixel(target_cell_y, center=True)
        cv2.line(board, (start_pixel_x, start_pixel_y), (target_pixel_x, target_pixel_y), color=color, thickness=thickness)

    def render(self):
        board = np.zeros((self.WIDTH, self.HEIGHT, 3))+255 # white background

        

        # render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y][x] == "#":
                    self._fill_cell(board, x, y, Colors.BLACK)
                    
        # render player
        self.player.draw(board)
        #self._fill_cell(board, self.player.x, self.player.y, self.player.color)

        # render agent
        self.agent.draw(board)
        # self._fill_cell(board, self.agent.x, self.agent.y, self.agent.color)

        
        see_color = Colors.GREEN if self.agent.is_seen else Colors.RED

        self._draw_grid(board)
        self._draw_line(board, self.player.x, self.player.y, self.agent.x, self.agent.y, see_color, 3)

        if self.mode == "human":
            cv2.imshow("Hide and Seek", board)

        return board

    def handle_inputs(self) -> bool:
        key = cv2.waitKey(int(1000/self.SPEED))

        # delete and escape keys
        if key in [8, 27]:
            return True
        
        new_pos = Vector2(self.player.pos.x, self.player.pos.y)
        if key == ord('q'):
            new_pos.x -= 1
        elif key == ord('d'):
            new_pos.x += 1
        elif key == ord('z'):
            new_pos.y -= 1
        elif key == ord('s'):
            new_pos.y += 1
        if key == ord('i'): # agent up
            self.handle_action(2)
        elif key == ord('j'): # agent left
            self.handle_action(0)
        elif key == ord('k'): # agent down
            self.handle_action(3)
        elif key == ord('l'): # agent right
            self.handle_action(1)
        elif key == ord('x'):
            return True
        
        if self._is_valid_coordinates(new_pos) and not self._is_wall(new_pos) and new_pos != self.agent.pos:
            self.player.pos = new_pos

        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False


        return False
    
    def _is_valid_coordinates(self, coord:Vector2) -> bool:
        return coord.x >= 0 and coord.x < self.GRID_W and coord.y >= 0 and coord.y < self.GRID_H

    def _is_wall(self, coord:Vector2) -> bool:
        return self.grid[coord.y][coord.x] == "#"

    # Handle action of the agent
    # The agent cannot move through walls and cannot move outside the grid
    # The action are:
    #   0: move left
    #   1: move right
    #   2: move up
    #   3: move down
    # Check first if the action is valid (not outside the grid, through a wall or on the player)
    # Then move the agent
    def handle_action(self, action) -> None:
        new_pos = Vector2(self.agent.pos.x, self.agent.pos.y)
        if action == 0:
            new_pos.x -= 1
        elif action == 1:
            new_pos.x += 1
        elif action == 2:
            new_pos.y -= 1
        elif action == 3:
            new_pos.y += 1
        
        if self._is_valid_coordinates(new_pos) and not self._is_wall(new_pos) and new_pos != self.player.pos:
            self.agent.pos = new_pos
        
        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False


    def run(self) -> None:
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
