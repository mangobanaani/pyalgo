"""
Basic Geometric Primitives

Fundamental geometric data structures and operations.
"""

import math
from typing import Tuple, List, Union
from dataclasses import dataclass

@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def distance_to_origin(self) -> float:
        """Calculate distance to origin."""
        return math.sqrt(self.x**2 + self.y**2)
    
    def dot_product(self, other: 'Point') -> float:
        """Calculate dot product with another point (treated as vector)."""
        return self.x * other.x + self.y * other.y
    
    def cross_product(self, other: 'Point') -> float:
        """Calculate cross product with another point (treated as vector)."""
        return self.x * other.y - self.y * other.x
    
    def normalize(self) -> 'Point':
        """Return a normalized version of this point (unit vector)."""
        magnitude = self.distance_to_origin()
        if magnitude == 0:
            return Point(0, 0)
        return self / magnitude
    
    def rotate(self, angle: float) -> 'Point':
        """Rotate point by angle (in radians) around origin."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Point(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

@dataclass
class Line:
    """Represents a line using two points."""
    p1: Point
    p2: Point
    
    def length(self) -> float:
        """Calculate the length of the line segment."""
        return self.p1.distance_to(self.p2)
    
    def midpoint(self) -> Point:
        """Calculate the midpoint of the line segment."""
        return Point((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)
    
    def slope(self) -> float:
        """Calculate the slope of the line."""
        if self.p2.x == self.p1.x:
            return float('inf')
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
    
    def y_intercept(self) -> float:
        """Calculate the y-intercept of the line."""
        slope = self.slope()
        if slope == float('inf'):
            return float('nan')
        return self.p1.y - slope * self.p1.x
    
    def point_at_parameter(self, t: float) -> Point:
        """Get point on line at parameter t (0 <= t <= 1 for segment)."""
        return self.p1 + (self.p2 - self.p1) * t
    
    def distance_to_point(self, point: Point) -> float:
        """Calculate the shortest distance from a point to this line."""
        # Vector from p1 to p2
        line_vec = self.p2 - self.p1
        # Vector from p1 to point
        point_vec = point - self.p1
        
        # Project point_vec onto line_vec
        line_length_sq = line_vec.x**2 + line_vec.y**2
        if line_length_sq == 0:
            return self.p1.distance_to(point)
        
        t = max(0, min(1, point_vec.dot_product(line_vec) / line_length_sq))
        projection = self.p1 + line_vec * t
        return point.distance_to(projection)

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return p1.distance_to(p2)

def cross_product(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate cross product of vectors (p2-p1) and (p3-p1).
    
    Returns:
        Positive if points are in counter-clockwise order,
        Negative if clockwise,
        Zero if collinear
    """
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

def orientation(p1: Point, p2: Point, p3: Point) -> int:
    """
    Find orientation of ordered triplet of points.
    
    Returns:
        0 -> Collinear points
        1 -> Clockwise orientation
        2 -> Counter-clockwise orientation
    """
    val = cross_product(p1, p2, p3)
    
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counter-clockwise

def polar_angle(point: Point, center: Point = Point(0, 0)) -> float:
    """Calculate polar angle of point relative to center."""
    return math.atan2(point.y - center.y, point.x - center.x)

def points_are_collinear(p1: Point, p2: Point, p3: Point, tolerance: float = 1e-10) -> bool:
    """Check if three points are collinear within tolerance."""
    return abs(cross_product(p1, p2, p3)) < tolerance

def triangle_area(p1: Point, p2: Point, p3: Point) -> float:
    """Calculate area of triangle formed by three points."""
    return abs(cross_product(p1, p2, p3)) / 2.0

def circumcenter(p1: Point, p2: Point, p3: Point) -> Point:
    """Calculate circumcenter of triangle formed by three points."""
    # Convert points to complex numbers for easier calculation
    z1 = complex(p1.x, p1.y)
    z2 = complex(p2.x, p2.y)
    z3 = complex(p3.x, p3.y)
    
    # Calculate circumcenter using complex arithmetic
    w = (z3 - z1) / (z2 - z1)
    if w.imag == 0:
        raise ValueError("Points are collinear")
    
    c = (z2 - z1) * (w - abs(w)**2) / (2j * w.imag) + z1
    return Point(c.real, c.imag)

def point_in_circle(point: Point, center: Point, radius: float) -> bool:
    """Check if point is inside circle."""
    return point.distance_to(center) <= radius

def line_circle_intersection(line: Line, center: Point, radius: float) -> List[Point]:
    """Find intersection points between line segment and circle."""
    # Vector from center to line start
    d = line.p1 - center
    # Line direction vector
    f = line.p2 - line.p1
    
    # Quadratic equation coefficients
    a = f.dot_product(f)
    b = 2 * f.dot_product(d)
    c = d.dot_product(d) - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return []  # No intersection
    
    intersections = []
    
    if discriminant == 0:
        # One intersection (tangent)
        t = -b / (2 * a)
        if 0 <= t <= 1:
            intersections.append(line.point_at_parameter(t))
    else:
        # Two intersections
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)
        
        if 0 <= t1 <= 1:
            intersections.append(line.point_at_parameter(t1))
        if 0 <= t2 <= 1:
            intersections.append(line.point_at_parameter(t2))
    
    return intersections
