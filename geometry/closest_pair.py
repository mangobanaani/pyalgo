"""
Closest Pair Algorithms

Algorithms for finding the closest pair of points in a set.
"""

import math
from typing import List, Tuple
from .basic_geometry import Point

def closest_pair_brute_force(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the closest pair of points using brute force approach.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    
    Args:
        points: List of points
        
    Returns:
        Tuple of (point1, point2, distance)
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points")
    
    min_distance = float('inf')
    closest_pair = (points[0], points[1])
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = points[i].distance_to(points[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])
    
    return closest_pair[0], closest_pair[1], min_distance

def closest_pair_divide_conquer(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the closest pair of points using divide and conquer approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        points: List of points
        
    Returns:
        Tuple of (point1, point2, distance)
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points")
    
    if len(points) <= 3:
        return closest_pair_brute_force(points)
    
    # Sort points by x-coordinate
    points_x = sorted(points, key=lambda p: p.x)
    # Sort points by y-coordinate
    points_y = sorted(points, key=lambda p: p.y)
    
    return _closest_pair_rec(points_x, points_y)

def _closest_pair_rec(px: List[Point], py: List[Point]) -> Tuple[Point, Point, float]:
    """
    Recursive helper function for divide and conquer closest pair.
    
    Args:
        px: Points sorted by x-coordinate
        py: Points sorted by y-coordinate
        
    Returns:
        Tuple of (point1, point2, distance)
    """
    n = len(px)
    
    # Base case: use brute force for small arrays
    if n <= 3:
        return closest_pair_brute_force(px)
    
    # Find the middle point
    mid = n // 2
    midpoint = px[mid]
    
    # Divide points in y sorted array around the vertical line
    pyl = [point for point in py if point.x < midpoint.x or 
           (point.x == midpoint.x and point.y < midpoint.y)]
    pyr = [point for point in py if point.x > midpoint.x or 
           (point.x == midpoint.x and point.y >= midpoint.y)]
    
    # Calculate the smallest distance on left and right recursively
    left_pair = _closest_pair_rec(px[:mid], pyl)
    right_pair = _closest_pair_rec(px[mid:], pyr)
    
    # Find the smaller of the two halves
    if left_pair[2] < right_pair[2]:
        min_distance = left_pair[2]
        closest_pair = (left_pair[0], left_pair[1])
    else:
        min_distance = right_pair[2]
        closest_pair = (right_pair[0], right_pair[1])
    
    # Find the closest split pair
    strip_pair = _closest_split_pair(px, py, midpoint.x, min_distance)
    
    if strip_pair[2] < min_distance:
        return strip_pair
    else:
        return closest_pair[0], closest_pair[1], min_distance

def _closest_split_pair(px: List[Point], py: List[Point], 
                       midx: float, delta: float) -> Tuple[Point, Point, float]:
    """
    Find the closest pair that spans the dividing line.
    
    Args:
        px: Points sorted by x-coordinate
        py: Points sorted by y-coordinate
        midx: x-coordinate of the dividing line
        delta: Current minimum distance
        
    Returns:
        Tuple of (point1, point2, distance)
    """
    # Create an array of points close to the line dividing the left and right halves
    strip = [point for point in py if abs(point.x - midx) < delta]
    
    min_distance = delta
    closest_pair = (px[0], px[1])  # Default pair
    
    # Find the closest points in strip
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j].y - strip[i].y) < min_distance:
            distance = strip[i].distance_to(strip[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (strip[i], strip[j])
            j += 1
    
    return closest_pair[0], closest_pair[1], min_distance

def all_pairs_distances(points: List[Point]) -> List[Tuple[Point, Point, float]]:
    """
    Calculate distances between all pairs of points.
    
    Args:
        points: List of points
        
    Returns:
        List of (point1, point2, distance) tuples sorted by distance
    """
    distances = []
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = points[i].distance_to(points[j])
            distances.append((points[i], points[j], distance))
    
    return sorted(distances, key=lambda x: x[2])

def k_closest_pairs(points: List[Point], k: int) -> List[Tuple[Point, Point, float]]:
    """
    Find the k closest pairs of points.
    
    Args:
        points: List of points
        k: Number of closest pairs to find
        
    Returns:
        List of k closest (point1, point2, distance) tuples
    """
    all_distances = all_pairs_distances(points)
    return all_distances[:k]

def farthest_pair_brute_force(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the farthest pair of points using brute force.
    
    Args:
        points: List of points
        
    Returns:
        Tuple of (point1, point2, distance)
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points")
    
    max_distance = 0
    farthest_pair = (points[0], points[1])
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = points[i].distance_to(points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (points[i], points[j])
    
    return farthest_pair[0], farthest_pair[1], max_distance

def diameter_of_point_set(points: List[Point]) -> float:
    """
    Find the diameter (maximum distance) of a set of points.
    
    Args:
        points: List of points
        
    Returns:
        Diameter of the point set
    """
    if len(points) < 2:
        return 0.0
    
    return farthest_pair_brute_force(points)[2]
