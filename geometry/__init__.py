"""
Computational Geometry algorithms package
Contains implementations of various geometric algorithms and data structures.
"""

from .basic_geometry import Point, Line, distance, cross_product, orientation
from .convex_hull import convex_hull_graham, convex_hull_jarvis
from .line_intersection import line_segment_intersection, line_line_intersection
from .polygon import polygon_area, point_in_polygon, polygon_centroid
from .closest_pair import closest_pair_brute_force, closest_pair_divide_conquer

__all__ = [
    'Point', 'Line', 'distance', 'cross_product', 'orientation',
    'convex_hull_graham', 'convex_hull_jarvis',
    'line_segment_intersection', 'line_line_intersection',
    'polygon_area', 'point_in_polygon', 'polygon_centroid',
    'closest_pair_brute_force', 'closest_pair_divide_conquer'
]
