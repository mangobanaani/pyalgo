"""
Line Intersection Algorithms

Algorithms for finding intersections between lines and line segments.
"""

from typing import Optional, Tuple
from .basic_geometry import Point, Line, cross_product

def line_line_intersection(line1: Line, line2: Line) -> Optional[Point]:
    """
    Find intersection point between two infinite lines.
    
    Args:
        line1: First line
        line2: Second line
        
    Returns:
        Intersection point or None if lines are parallel
    """
    x1, y1 = line1.p1.x, line1.p1.y
    x2, y2 = line1.p2.x, line1.p2.y
    x3, y3 = line2.p1.x, line2.p1.y
    x4, y4 = line2.p2.x, line2.p2.y
    
    # Calculate the denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Lines are parallel
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return Point(x, y)

def line_segment_intersection(seg1: Line, seg2: Line) -> Optional[Point]:
    """
    Find intersection point between two line segments.
    
    Args:
        seg1: First line segment
        seg2: Second line segment
        
    Returns:
        Intersection point or None if segments don't intersect
    """
    p1, q1 = seg1.p1, seg1.p2
    p2, q2 = seg2.p1, seg2.p2
    
    # Find the four orientations needed for general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return line_line_intersection(seg1, seg2)
    
    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return p2
    
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return q2
    
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return p1
    
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return q1
    
    return None

def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Find orientation of ordered triplet (p, q, r).
    
    Returns:
        0 -> p, q and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    
    if abs(val) < 1e-10:
        return 0  # Collinear
    
    return 1 if val > 0 else 2

def on_segment(p: Point, q: Point, r: Point) -> bool:
    """
    Check if point q lies on line segment pr.
    (Given that p, q, r are collinear)
    """
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

def segments_intersect(seg1: Line, seg2: Line) -> bool:
    """
    Check if two line segments intersect.
    
    Args:
        seg1: First line segment
        seg2: Second line segment
        
    Returns:
        True if segments intersect
    """
    return line_segment_intersection(seg1, seg2) is not None

def point_line_distance(point: Point, line: Line) -> float:
    """
    Calculate the shortest distance from a point to a line.
    
    Args:
        point: Point
        line: Line
        
    Returns:
        Shortest distance
    """
    return line.distance_to_point(point)

def point_segment_distance(point: Point, segment: Line) -> float:
    """
    Calculate the shortest distance from a point to a line segment.
    
    Args:
        point: Point
        segment: Line segment
        
    Returns:
        Shortest distance
    """
    return segment.distance_to_point(point)

def segment_segment_distance(seg1: Line, seg2: Line) -> float:
    """
    Calculate the shortest distance between two line segments.
    
    Args:
        seg1: First line segment
        seg2: Second line segment
        
    Returns:
        Shortest distance between segments
    """
    # Check if segments intersect
    if segments_intersect(seg1, seg2):
        return 0.0
    
    # Calculate distances from endpoints to segments
    distances = [
        point_segment_distance(seg1.p1, seg2),
        point_segment_distance(seg1.p2, seg2),
        point_segment_distance(seg2.p1, seg1),
        point_segment_distance(seg2.p2, seg1)
    ]
    
    return min(distances)

def ray_segment_intersection(ray_origin: Point, ray_direction: Point, segment: Line) -> Optional[Point]:
    """
    Find intersection between a ray and a line segment.
    
    Args:
        ray_origin: Starting point of the ray
        ray_direction: Direction vector of the ray
        segment: Line segment
        
    Returns:
        Intersection point or None if no intersection
    """
    # Convert ray to line segment (with a very far endpoint)
    far_point = ray_origin + ray_direction * 1e6
    ray_line = Line(ray_origin, far_point)
    
    intersection = line_segment_intersection(ray_line, segment)
    
    if intersection is None:
        return None
    
    # Check if intersection is in the direction of the ray
    to_intersection = intersection - ray_origin
    if to_intersection.dot_product(ray_direction) >= 0:
        return intersection
    
    return None
