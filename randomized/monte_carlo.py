"""
Monte Carlo methods and algorithms.

Monte Carlo methods are computational algorithms that rely on repeated
random sampling to obtain numerical results. They are useful for solving
problems that might be deterministic in principle.
"""

import random
import math
from typing import Callable, List, Tuple


def monte_carlo_pi(num_samples: int) -> float:
    """
    Estimate π using Monte Carlo method.
    
    Uses the ratio of points inside a unit circle to total points
    in a unit square to estimate π.
    
    Args:
        num_samples: Number of random points to generate
        
    Returns:
        Estimated value of π
        
    Time Complexity: O(num_samples)
    Space Complexity: O(1)
    """
    points_inside_circle = 0
    
    for _ in range(num_samples):
        # Generate random point in unit square [-1, 1] x [-1, 1]
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # Check if point is inside unit circle
        if x * x + y * y <= 1:
            points_inside_circle += 1
    
    # π ≈ 4 * (points inside circle / total points)
    return 4.0 * points_inside_circle / num_samples


def monte_carlo_integration(func: Callable[[float], float], 
                           a: float, b: float, 
                           num_samples: int) -> float:
    """
    Estimate definite integral using Monte Carlo method.
    
    Estimates ∫[a,b] f(x) dx using random sampling.
    
    Args:
        func: Function to integrate
        a: Lower bound of integration
        b: Upper bound of integration
        num_samples: Number of random samples
        
    Returns:
        Estimated value of the integral
        
    Time Complexity: O(num_samples)
    Space Complexity: O(1)
    """
    if a >= b:
        return 0.0
    
    total = 0.0
    
    for _ in range(num_samples):
        # Generate random point in interval [a, b]
        x = random.uniform(a, b)
        total += func(x)
    
    # Integral ≈ (b - a) * average function value
    return (b - a) * total / num_samples


def monte_carlo_integration_2d(func: Callable[[float, float], float],
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float],
                              num_samples: int) -> float:
    """
    Estimate 2D integral using Monte Carlo method.
    
    Estimates ∫∫[R] f(x,y) dx dy over rectangle R.
    
    Args:
        func: Function f(x,y) to integrate
        x_range: (x_min, x_max) bounds for x
        y_range: (y_min, y_max) bounds for y
        num_samples: Number of random samples
        
    Returns:
        Estimated value of the 2D integral
        
    Time Complexity: O(num_samples)
    Space Complexity: O(1)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    if x_min >= x_max or y_min >= y_max:
        return 0.0
    
    total = 0.0
    
    for _ in range(num_samples):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        total += func(x, y)
    
    # Integral ≈ area * average function value
    area = (x_max - x_min) * (y_max - y_min)
    return area * total / num_samples


def monte_carlo_option_pricing(S0: float, K: float, T: float, 
                              r: float, sigma: float, 
                              num_simulations: int,
                              option_type: str = 'call') -> float:
    """
    Price European options using Monte Carlo simulation.
    
    Uses geometric Brownian motion to simulate stock price paths
    and calculate option payoffs.
    
    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        num_simulations: Number of simulation paths
        option_type: 'call' or 'put'
        
    Returns:
        Estimated option price
        
    Time Complexity: O(num_simulations)
    Space Complexity: O(1)
    """
    payoff_sum = 0.0
    
    for _ in range(num_simulations):
        # Generate random normal variable
        Z = random.gauss(0, 1)
        
        # Calculate stock price at maturity using Black-Scholes formula
        ST = S0 * math.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
        
        # Calculate payoff
        if option_type.lower() == 'call':
            payoff = max(ST - K, 0)
        elif option_type.lower() == 'put':
            payoff = max(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        payoff_sum += payoff
    
    # Discount average payoff to present value
    return math.exp(-r * T) * payoff_sum / num_simulations


def monte_carlo_area_estimation(vertices: List[Tuple[float, float]], 
                               num_samples: int) -> float:
    """
    Estimate area of a polygon using Monte Carlo method.
    
    Args:
        vertices: List of (x, y) coordinates defining polygon vertices
        num_samples: Number of random points to test
        
    Returns:
        Estimated area of the polygon
        
    Time Complexity: O(num_samples * n) where n is number of vertices
    Space Complexity: O(1)
    """
    if len(vertices) < 3:
        return 0.0
    
    # Find bounding box
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    if x_min >= x_max or y_min >= y_max:
        return 0.0
    
    points_inside = 0
    
    for _ in range(num_samples):
        # Generate random point in bounding box
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        # Check if point is inside polygon using ray casting
        if _point_in_polygon(x, y, vertices):
            points_inside += 1
    
    # Area ≈ bounding box area * (points inside / total points)
    bounding_box_area = (x_max - x_min) * (y_max - y_min)
    return bounding_box_area * points_inside / num_samples


def _point_in_polygon(x: float, y: float, 
                      vertices: List[Tuple[float, float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    
    Args:
        x, y: Point coordinates
        vertices: Polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def monte_carlo_variance_reduction(func: Callable[[float], float],
                                  control_func: Callable[[float], float],
                                  control_mean: float,
                                  a: float, b: float,
                                  num_samples: int) -> float:
    """
    Monte Carlo integration with control variates for variance reduction.
    
    Uses a control variate (function with known expected value) to reduce
    the variance of the Monte Carlo estimate.
    
    Args:
        func: Function to integrate
        control_func: Control function with known mean
        control_mean: Known expected value of control function
        a: Lower bound
        b: Upper bound
        num_samples: Number of samples
        
    Returns:
        Improved estimate with reduced variance
    """
    if num_samples < 2:
        return monte_carlo_integration(func, a, b, num_samples)
    
    func_values = []
    control_values = []
    
    # Collect samples
    for _ in range(num_samples):
        x = random.uniform(a, b)
        func_values.append(func(x))
        control_values.append(control_func(x))
    
    # Calculate means
    func_mean = sum(func_values) / num_samples
    control_sample_mean = sum(control_values) / num_samples
    
    # Calculate covariance and control variate coefficient
    covariance = sum((f - func_mean) * (c - control_sample_mean) 
                    for f, c in zip(func_values, control_values)) / (num_samples - 1)
    control_variance = sum((c - control_sample_mean) ** 2 
                          for c in control_values) / (num_samples - 1)
    
    if control_variance == 0:
        beta = 0
    else:
        beta = covariance / control_variance
    
    # Apply control variate correction
    corrected_mean = func_mean - beta * (control_sample_mean - control_mean)
    
    return (b - a) * corrected_mean


def monte_carlo_importance_sampling(func: Callable[[float], float],
                                   proposal_sampler: Callable[[], float],
                                   proposal_pdf: Callable[[float], float],
                                   target_pdf: Callable[[float], float],
                                   num_samples: int) -> float:
    """
    Monte Carlo estimation using importance sampling.
    
    Uses a proposal distribution to sample from regions of higher importance.
    
    Args:
        func: Function to estimate expectation of
        proposal_sampler: Function that generates samples from proposal distribution
        proposal_pdf: PDF of proposal distribution
        target_pdf: PDF of target distribution
        num_samples: Number of samples
        
    Returns:
        Importance sampling estimate
    """
    total = 0.0
    
    for _ in range(num_samples):
        # Sample from proposal distribution
        x = proposal_sampler()
        
        # Calculate importance weight
        if proposal_pdf(x) > 0:
            weight = target_pdf(x) / proposal_pdf(x)
            total += func(x) * weight
    
    return total / num_samples
