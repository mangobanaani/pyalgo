"""
Polygon Algorithms

Algorithms for working with polygons.
"""

import math
from typing import List
from .basic_geometry import Point, cross_product

def polygon_area(vertices: List[Point]) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        Area of the polygon
    """
    if len(vertices) < 3:
        return 0.0
    
    area = 0.0
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y
    
    return abs(area) / 2.0

def point_in_polygon(point: Point, vertices: List[Point]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point to check
        vertices: List of polygon vertices in order
        
    Returns:
        True if point is inside the polygon
    """
    if len(vertices) < 3:
        return False
    
    x, y = point.x, point.y
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0].x, vertices[0].y
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n].x, vertices[i % n].y
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside

def polygon_centroid(vertices: List[Point]) -> Point:
    """
    Calculate the centroid of a polygon.
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        Centroid point
    """
    if len(vertices) < 3:
        if len(vertices) == 1:
            return vertices[0]
        elif len(vertices) == 2:
            return Point((vertices[0].x + vertices[1].x) / 2, 
                        (vertices[0].y + vertices[1].y) / 2)
        else:
            return Point(0, 0)
    
    area = polygon_area(vertices)
    if area == 0:
        # Degenerate polygon, return geometric center
        cx = sum(v.x for v in vertices) / len(vertices)
        cy = sum(v.y for v in vertices) / len(vertices)
        return Point(cx, cy)
    
    cx = 0.0
    cy = 0.0
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y
        cx += (vertices[i].x + vertices[j].x) * cross
        cy += (vertices[i].y + vertices[j].y) * cross
    
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    
    return Point(cx, cy)

def polygon_perimeter(vertices: List[Point]) -> float:
    """
    Calculate the perimeter of a polygon.
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        Perimeter of the polygon
    """
    if len(vertices) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        perimeter += vertices[i].distance_to(vertices[j])
    
    return perimeter

def is_polygon_convex(vertices: List[Point]) -> bool:
    """
    Check if a polygon is convex.
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        True if polygon is convex
    """
    if len(vertices) < 3:
        return True
    
    n = len(vertices)
    sign = None
    
    for i in range(n):
        j = (i + 1) % n
        k = (i + 2) % n
        
        cross = cross_product(vertices[i], vertices[j], vertices[k])
        
        if abs(cross) < 1e-10:
            continue  # Skip collinear points
        
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    
    return True

def polygon_orientation(vertices: List[Point]) -> int:
    """
    Determine the orientation of a polygon.
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        1 for counter-clockwise, -1 for clockwise, 0 for degenerate
    """
    if len(vertices) < 3:
        return 0
    
    # Find the signed area
    signed_area = 0.0
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        signed_area += (vertices[j].x - vertices[i].x) * (vertices[j].y + vertices[i].y)
    
    if abs(signed_area) < 1e-10:
        return 0  # Degenerate
    
    return -1 if signed_area > 0 else 1  # Clockwise if positive

def simplify_polygon(vertices: List[Point], tolerance: float = 1e-10) -> List[Point]:
    """
    Simplify a polygon by removing collinear points.
    
    Args:
        vertices: List of polygon vertices
        tolerance: Tolerance for collinearity check
        
    Returns:
        Simplified polygon vertices
    """
    if len(vertices) < 3:
        return vertices.copy()
    
    simplified = []
    n = len(vertices)
    
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        
        # Check if current point is collinear with its neighbors
        cross = cross_product(vertices[prev_idx], vertices[i], vertices[next_idx])
        
        if abs(cross) > tolerance:
            simplified.append(vertices[i])
    
    return simplified

def point_polygon_distance(point: Point, vertices: List[Point]) -> float:
    """
    Calculate the shortest distance from a point to a polygon.
    
    Args:
        point: Point
        vertices: List of polygon vertices
        
    Returns:
        Shortest distance to the polygon
    """
    if len(vertices) < 3:
        return float('inf')
    
    if point_in_polygon(point, vertices):
        return 0.0
    
    min_distance = float('inf')
    n = len(vertices)
    
    # Check distance to each edge
    for i in range(n):
        j = (i + 1) % n
        edge_start = vertices[i]
        edge_end = vertices[j]
        
        # Distance to line segment
        edge_vec = edge_end - edge_start
        point_vec = point - edge_start
        
        edge_length_sq = edge_vec.x**2 + edge_vec.y**2
        if edge_length_sq == 0:
            dist = edge_start.distance_to(point)
        else:
            t = max(0, min(1, point_vec.dot_product(edge_vec) / edge_length_sq))
            projection = edge_start + edge_vec * t
            dist = point.distance_to(projection)
        
        min_distance = min(min_distance, dist)
    
    return min_distance

def triangulate_polygon_ear_clipping(vertices: List[Point]) -> List[List[Point]]:
    """
    Triangulate a simple polygon using ear clipping algorithm.
    
    Args:
        vertices: List of polygon vertices in counter-clockwise order
        
    Returns:
        List of triangles (each triangle is a list of 3 points)
    """
    if len(vertices) < 3:
        return []
    
    if len(vertices) == 3:
        return [vertices.copy()]
    
    triangles = []
    remaining = vertices.copy()
    
    while len(remaining) > 3:
        ear_found = False
        
        for i in range(len(remaining)):
            if is_ear(remaining, i):
                # Create triangle
                prev_idx = (i - 1) % len(remaining)
                next_idx = (i + 1) % len(remaining)
                
                triangle = [remaining[prev_idx], remaining[i], remaining[next_idx]]
                triangles.append(triangle)
                
                # Remove the ear vertex
                remaining.pop(i)
                ear_found = True
                break
        
        if not ear_found:
            # Fallback: create triangle with first three vertices
            triangles.append(remaining[:3])
            remaining = remaining[2:]
    
    if len(remaining) == 3:
        triangles.append(remaining)
    
    return triangles

def is_ear(vertices: List[Point], index: int) -> bool:
    """
    Check if a vertex is an ear (can be safely removed in triangulation).
    
    Args:
        vertices: List of polygon vertices
        index: Index of vertex to check
        
    Returns:
        True if vertex is an ear
    """
    n = len(vertices)
    if n < 3:
        return False
    
    prev_idx = (index - 1) % n
    next_idx = (index + 1) % n
    
    # Check if the triangle is oriented correctly (counter-clockwise)
    if cross_product(vertices[prev_idx], vertices[index], vertices[next_idx]) <= 0:
        return False
    
    # Check if any other vertex is inside this triangle
    for i in range(n):
        if i == prev_idx or i == index or i == next_idx:
            continue
        
        if point_in_triangle(vertices[i], vertices[prev_idx], vertices[index], vertices[next_idx]):
            return False
    
    return True

def point_in_triangle(point: Point, a: Point, b: Point, c: Point) -> bool:
    """
    Check if a point is inside a triangle using barycentric coordinates.
    
    Args:
        point: Point to check
        a, b, c: Triangle vertices
        
    Returns:
        True if point is inside the triangle
    """
    # Calculate barycentric coordinates
    denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
    
    if abs(denom) < 1e-10:
        return False  # Degenerate triangle
    
    alpha = ((b.y - c.y) * (point.x - c.x) + (c.x - b.x) * (point.y - c.y)) / denom
    beta = ((c.y - a.y) * (point.x - c.x) + (a.x - c.x) * (point.y - c.y)) / denom
    gamma = 1 - alpha - beta
    
    return alpha >= 0 and beta >= 0 and gamma >= 0
