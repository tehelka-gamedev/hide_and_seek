from __future__ import annotations
import cv2

import numpy as np
from Vector2 import Vector2, lerp_vec2

import Colors

class Entity:
    def __init__(self, pos=Vector2(-1,-1), color=Colors.GREY) -> None:
        self.pos = pos.floor()
        self.color = color
        self.is_seen = False

    @property
    def x(self) -> int:
        return self.pos.x
    
    @property
    def y(self) -> int:
        return self.pos.y
    
    @x.setter
    def x(self, value:int) -> None:
        self.pos.x = value

    @y.setter
    def y(self, value:int) -> None:
        self.pos.y = value
    
    def move(self, dir:Vector2) -> None:
        self.pos += dir

    def draw(self, board) -> None:
        CELL_SIZE = 32
        pixel = self.pos * CELL_SIZE + CELL_SIZE//2
        cv2.circle(board, (pixel.x, pixel.y), CELL_SIZE//2-1, self.color, -1)
    

    def can_see(self, target:Entity, grid) -> bool:
        """
        An target entity e2 can be seen from an entity e1 if the ray from e1 to e2 does not intersect any wall
        Returns True if the target is seen, False otherwise

        Sample n+1 points on the grid between e1 and e2 where N is the diagonal distance between e1 and e2
        The n+1 interpolation points are evenly spaced
        Those points are rounded to the nearest grid cell.
        (see https://www.redblobgames.com/grids/line-drawing.html)

        If one of the points is a wall, the target is not seen
        """

        
        n = self.pos.diagonal_distance_to(target.pos)
        for step in range(n+1):
            t = 0.0 if n == 0 else step / n
            point_in_grid = lerp_vec2(self.pos, target.pos, t).round()
            if grid[point_in_grid.y][point_in_grid.x] == "#":
                return False
            
        
        # no obstacle!
        return True

