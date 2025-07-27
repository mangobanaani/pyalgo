"""
Convex Hull Algorithms

Algorithms for finding the convex hull of a set of points.
"""

import math
from typing import List
from .basic_geometry import Point, cross_product, polar_angle

def convex_hull_graham(points: List[Point]) -> List[Point]:
    """
    Graham scan algorithm for finding convex hull.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull in counter-clockwise order
    """
    if len(points) < 3:
        return points.copy()
    
    # Find the bottom-most point (and leftmost in case of tie)
    start = min(points, key=lambda p: (p.y, p.x))
    
    # Sort points by polar angle with respect to start point
    def compare_points(p1, p2):
        angle1 = polar_angle(p1, start)
        angle2 = polar_angle(p2, start)
        
        if angle1 != angle2:
            return angle1 - angle2
        
        # If angles are equal, closer point comes first
        dist1 = start.distance_to(p1)
        dist2 = start.distance_to(p2)
        return dist1 - dist2
    
    # Remove start point and sort others
    other_points = [p for p in points if p != start]
    other_points.sort(key=lambda p: (polar_angle(p, start), start.distance_to(p)))
    
    # Create hull using stack
    hull = [start]
    
    for point in other_points:
        # Remove points that create clockwise turn
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], point) < 0:
            hull.pop()
        hull.append(point)
    
    return hull

def convex_hull_jarvis(points: List[Point]) -> List[Point]:
    """
    Jarvis march (Gift wrapping) algorithm for finding convex hull.
    
    Time Complexity: O(nh) where h is the number of hull points
    Space Complexity: O(h)
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull in counter-clockwise order
    """
    if len(points) < 3:
        return points.copy()
    
    n = len(points)
    hull = []
    
    # Find the leftmost point
    leftmost = min(range(n), key=lambda i: (points[i].x, points[i].y))
    
    current = leftmost
    while True:
        hull.append(points[current])
        
        # Find the most counter-clockwise point
        next_point = (current + 1) % n
        
        for i in range(n):
            if i == current:
                continue
                
            # If i is more counter-clockwise than next_point
            cross = cross_product(points[current], points[next_point], points[i])
            if cross > 0:
                next_point = i
            elif cross == 0:
                # If collinear, choose the farther point
                dist_next = points[current].distance_to(points[next_point])
                dist_i = points[current].distance_to(points[i])
                if dist_i > dist_next:
                    next_point = i
        
        current = next_point
        
        # If we've come back to the start, we're done
        if current == leftmost:
            break
    
    return hull

def convex_hull_quickhull(points: List[Point]) -> List[Point]:
    """
    QuickHull algorithm for finding convex hull.
    
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(n)
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull
    """
    if len(points) < 3:
        return points.copy()
    
    # Find the leftmost and rightmost points
    min_x = min(points, key=lambda p: p.x)
    max_x = max(points, key=lambda p: p.x)
    
    hull = []
    
    def find_hull(p1: Point, p2: Point, point_set: List[Point]):
        """Recursive function to find hull points."""
        if not point_set:
            return
        
        # Find the point farthest from line p1-p2
        max_dist = 0
        farthest_point = None
        
        for point in point_set:
            dist = abs(cross_product(p1, p2, point))
            if dist > max_dist:
                max_dist = dist
                farthest_point = point
        
        if farthest_point is None:
            return
        
        hull.append(farthest_point)
        
        # Divide the remaining points
        left_set = []
        right_set = []
        
        for point in point_set:
            if point == farthest_point:
                continue
            
            if cross_product(p1, farthest_point, point) > 0:
                left_set.append(point)
            elif cross_product(farthest_point, p2, point) > 0:
                right_set.append(point)
        
        find_hull(p1, farthest_point, left_set)
        find_hull(farthest_point, p2, right_set)
    
    # Divide points into upper and lower sets
    upper_set = []
    lower_set = []
    
    for point in points:
        if point == min_x or point == max_x:
            continue
        
        cross = cross_product(min_x, max_x, point)
        if cross > 0:
            upper_set.append(point)
        elif cross < 0:
            lower_set.append(point)
    
    hull.append(min_x)
    find_hull(min_x, max_x, upper_set)
    hull.append(max_x)
    find_hull(max_x, min_x, lower_set)
    
    return hull

def convex_hull_andrew(points: List[Point]) -> List[Point]:
    """
    Andrew's monotone chain algorithm for finding convex hull.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull
    """
    if len(points) < 3:
        return points.copy()
    
    # Sort points lexicographically
    sorted_points = sorted(points, key=lambda p: (p.x, p.y))
    
    # Build lower hull
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(sorted_points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]

def is_point_inside_convex_polygon(point: Point, hull: List[Point]) -> bool:
    """
    Check if a point is inside a convex polygon.
    
    Args:
        point: Point to check
        hull: Convex hull points in counter-clockwise order
        
    Returns:
        True if point is inside the polygon
    """
    if len(hull) < 3:
        return False
    
    # Check if point is on the same side of all edges
    sign = None
    
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        cross = cross_product(hull[i], hull[j], point)
        
        if cross == 0:
            # Point is on the edge
            return True
        
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    
    return True
