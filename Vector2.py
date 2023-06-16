from __future__ import annotations

class Vector2:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y
    
    # function that adds two vectors or a vector and a scalar
    def __add__(self, other) -> Vector2:
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2(self.x + other, self.y + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'Vector2' and '{type(other)}'")

    # function that subtracts two vectors or a vector and a scalar
    def __sub__(self, other) -> Vector2:
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2(self.x - other, self.y - other)
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'Vector2' and '{type(other)}'")
    
    # function that multiplies this vector by a scalar
    def __mul__(self, other:float) -> Vector2:
        return Vector2(self.x * other, self.y * other)
    
    # function that divides this vector by a scalar
    def __truediv__(self, other:float) -> Vector2:
        return Vector2(self.x / other, self.y / other)
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    # round down the coordinates
    def floor(self) -> Vector2:
        return Vector2(int(self.x), int(self.y))
    
    def __eq__(self, other:Vector2) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other:Vector2) -> bool:
        return not self.__eq__(other)
    
    def manhattan_distance(self, other:Vector2) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def diagonal_distance_to(self, other:Vector2) -> int:
        return diagonal_distance(self, other)
    
    def round(self) -> Vector2:
        return Vector2(round(self.x), round(self.y))


def diagonal_distance(start:Vector2, end:Vector2) -> int:
    """
    Returns the diagonal distance between two points
    """
    return max(abs(start.x - end.x), abs(start.y - end.y))


def lerp_vec2(start:Vector2, end:Vector2, t:float) -> Vector2:
    """
    Linear interpolation between two points
    """
    return start + (end - start) * t