"""
Test cases for Geometry algorithms
"""

import pytest
import math
from geometry.basic_geometry import Point, Line, distance, cross_product, orientation
from geometry.convex_hull import convex_hull_graham, convex_hull_jarvis
from geometry.line_intersection import line_segment_intersection, line_line_intersection
from geometry.polygon import polygon_area, point_in_polygon, polygon_centroid
from geometry.closest_pair import closest_pair_brute_force, closest_pair_divide_conquer

class TestBasicGeometry:
    def test_point_operations(self):
        """Test basic point operations."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        
        # Test distance
        assert p1.distance_to(p2) == 5.0
        assert p2.distance_to_origin() == 5.0
        
        # Test arithmetic operations
        p3 = p1 + p2
        assert p3.x == 3 and p3.y == 4
        
        p4 = p2 - p1
        assert p4.x == 3 and p4.y == 4
        
        p5 = p2 * 2
        assert p5.x == 6 and p5.y == 8
    
    def test_cross_product(self):
        """Test cross product calculation."""
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(0, 1)
        
        # Counter-clockwise
        assert cross_product(p1, p2, p3) > 0
        
        # Clockwise
        assert cross_product(p1, p3, p2) < 0
        
        # Collinear
        p4 = Point(2, 0)
        assert cross_product(p1, p2, p4) == 0
    
    def test_line_operations(self):
        """Test line operations."""
        line = Line(Point(0, 0), Point(3, 4))
        
        assert line.length() == 5.0
        midpoint = line.midpoint()
        assert midpoint.x == 1.5 and midpoint.y == 2.0
        
        # Test point at parameter
        point = line.point_at_parameter(0.5)
        assert point.x == 1.5 and point.y == 2.0

class TestConvexHull:
    def test_convex_hull_graham(self):
        """Test Graham scan convex hull."""
        points = [
            Point(0, 0), Point(1, 0), Point(2, 0), Point(1, 1),
            Point(0, 1), Point(0.5, 0.5)
        ]
        
        hull = convex_hull_graham(points)
        
        # Should have 4 points on the hull
        assert len(hull) == 4
        
        # Check that all hull points are from original set
        for point in hull:
            assert point in points
    
    def test_convex_hull_jarvis(self):
        """Test Jarvis march convex hull."""
        points = [
            Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 2), Point(2, 2)
        ]
        
        hull = convex_hull_jarvis(points)
        
        # Should have square vertices
        assert len(hull) == 4
    
    def test_convex_hull_empty(self):
        """Test convex hull with few points."""
        # Single point
        assert convex_hull_graham([Point(0, 0)]) == [Point(0, 0)]
        
        # Two points
        points = [Point(0, 0), Point(1, 1)]
        assert len(convex_hull_graham(points)) == 2

class TestLineIntersection:
    def test_line_line_intersection(self):
        """Test line-line intersection."""
        line1 = Line(Point(0, 0), Point(2, 2))
        line2 = Line(Point(0, 2), Point(2, 0))
        
        intersection = line_line_intersection(line1, line2)
        
        assert intersection is not None
        assert abs(intersection.x - 1.0) < 1e-10
        assert abs(intersection.y - 1.0) < 1e-10
    
    def test_parallel_lines(self):
        """Test parallel lines intersection."""
        line1 = Line(Point(0, 0), Point(1, 0))
        line2 = Line(Point(0, 1), Point(1, 1))
        
        intersection = line_line_intersection(line1, line2)
        assert intersection is None
    
    def test_line_segment_intersection(self):
        """Test line segment intersection."""
        seg1 = Line(Point(0, 0), Point(2, 2))
        seg2 = Line(Point(0, 2), Point(2, 0))
        
        intersection = line_segment_intersection(seg1, seg2)
        
        assert intersection is not None
        assert abs(intersection.x - 1.0) < 1e-10
        assert abs(intersection.y - 1.0) < 1e-10

class TestPolygon:
    def test_polygon_area(self):
        """Test polygon area calculation."""
        # Unit square
        square = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
        assert polygon_area(square) == 1.0
        
        # Triangle
        triangle = [Point(0, 0), Point(1, 0), Point(0.5, 1)]
        assert polygon_area(triangle) == 0.5
    
    def test_point_in_polygon(self):
        """Test point in polygon detection."""
        # Unit square
        square = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
        
        # Point inside
        assert point_in_polygon(Point(0.5, 0.5), square)
        
        # Point outside
        assert not point_in_polygon(Point(2, 2), square)
        
        # Point on edge (might be implementation dependent)
        # assert point_in_polygon(Point(0.5, 0), square)
    
    def test_polygon_centroid(self):
        """Test polygon centroid calculation."""
        # Unit square
        square = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
        centroid = polygon_centroid(square)
        
        assert abs(centroid.x - 0.5) < 1e-10
        assert abs(centroid.y - 0.5) < 1e-10

class TestClosestPair:
    def test_closest_pair_brute_force(self):
        """Test brute force closest pair."""
        points = [Point(0, 0), Point(1, 1), Point(3, 3), Point(1, 0)]
        
        p1, p2, dist = closest_pair_brute_force(points)
        
        # Closest should be (0,0) and (1,0) with distance 1
        assert dist == 1.0
        assert {(p1.x, p1.y), (p2.x, p2.y)} == {(0, 0), (1, 0)}
    
    def test_closest_pair_divide_conquer(self):
        """Test divide and conquer closest pair."""
        points = [Point(0, 0), Point(1, 1), Point(3, 3), Point(1, 0)]
        
        p1, p2, dist = closest_pair_divide_conquer(points)
        
        # Should give same result as brute force
        assert dist == 1.0
        assert {(p1.x, p1.y), (p2.x, p2.y)} == {(0, 0), (1, 0)}
    
    def test_closest_pair_large(self):
        """Test closest pair with larger input."""
        points = [Point(i, i) for i in range(10)]
        points.append(Point(5.1, 5.1))  # Very close to Point(5, 5)
        
        p1, p2, dist = closest_pair_divide_conquer(points)
        
        # Should find the closest pair
        assert dist < 0.2  # Should be about 0.14
    
    def test_closest_pair_same_result(self):
        """Test that both algorithms give same result."""
        points = [Point(0, 0), Point(3, 4), Point(1, 1), Point(2, 3), Point(5, 1)]
        
        bf_result = closest_pair_brute_force(points)
        dc_result = closest_pair_divide_conquer(points)
        
        assert abs(bf_result[2] - dc_result[2]) < 1e-10
