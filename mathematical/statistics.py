"""
Statistics and probability algorithms.

This module implements statistical measures, distributions,
and hypothesis testing methods.
"""

from typing import List, Tuple, Optional, Dict
import math


def mean(data: List[float]) -> float:
    """
    Calculate arithmetic mean.
    
    Args:
        data: List of numerical values
        
    Returns:
        Arithmetic mean
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """
    Calculate median value.
    
    Args:
        data: List of numerical values
        
    Returns:
        Median value
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not data:
        raise ValueError("Cannot calculate median of empty list")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]


def mode(data: List[float]) -> List[float]:
    """
    Calculate mode(s) - most frequent value(s).
    
    Args:
        data: List of numerical values
        
    Returns:
        List of mode values
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not data:
        raise ValueError("Cannot calculate mode of empty list")
    
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    
    max_freq = max(frequency.values())
    modes = [value for value, freq in frequency.items() if freq == max_freq]
    
    return modes


def variance(data: List[float], population: bool = False) -> float:
    """
    Calculate variance.
    
    Args:
        data: List of numerical values
        population: If True, calculate population variance; if False, sample variance
        
    Returns:
        Variance
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not data:
        raise ValueError("Cannot calculate variance of empty list")
    
    if len(data) == 1 and not population:
        raise ValueError("Cannot calculate sample variance with only one data point")
    
    data_mean = mean(data)
    sum_squared_diff = sum((x - data_mean) ** 2 for x in data)
    
    divisor = len(data) if population else len(data) - 1
    return sum_squared_diff / divisor


def standard_deviation(data: List[float], population: bool = False) -> float:
    """
    Calculate standard deviation.
    
    Args:
        data: List of numerical values
        population: If True, calculate population std; if False, sample std
        
    Returns:
        Standard deviation
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    return math.sqrt(variance(data, population))


def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        x: First variable data
        y: Second variable data
        
    Returns:
        Correlation coefficient (-1 to 1)
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(x) != len(y):
        raise ValueError("Both variables must have same length")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 data points")
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0  # No correlation if either variable has no variance
    
    return numerator / denominator


def covariance(x: List[float], y: List[float], population: bool = False) -> float:
    """
    Calculate covariance between two variables.
    
    Args:
        x: First variable data
        y: Second variable data
        population: If True, population covariance; if False, sample covariance
        
    Returns:
        Covariance
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(x) != len(y):
        raise ValueError("Both variables must have same length")
    
    if len(x) == 0:
        raise ValueError("Cannot calculate covariance of empty lists")
    
    if len(x) == 1 and not population:
        raise ValueError("Cannot calculate sample covariance with only one data point")
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    sum_products = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    divisor = n if population else n - 1
    return sum_products / divisor


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Perform simple linear regression y = a + bx.
    
    Args:
        x: Independent variable data
        y: Dependent variable data
        
    Returns:
        Tuple (slope, intercept, r_squared)
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(x) != len(y):
        raise ValueError("Both variables must have same length")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 data points for regression")
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Calculate slope
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    
    if denominator == 0:
        raise ValueError("Cannot perform regression: no variance in x")
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    # Calculate R-squared
    y_pred = [intercept + slope * x[i] for i in range(n)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
    
    return slope, intercept, r_squared


def normal_distribution(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate normal distribution probability density.
    
    Args:
        x: Value to evaluate
        mu: Mean
        sigma: Standard deviation
        
    Returns:
        Probability density at x
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    
    return coefficient * math.exp(exponent)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate normal distribution cumulative distribution function.
    
    Uses approximation for standard normal CDF.
    
    Args:
        x: Value to evaluate
        mu: Mean
        sigma: Standard deviation
        
    Returns:
        Cumulative probability
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # Standardize
    z = (x - mu) / sigma
    
    # Approximation for standard normal CDF
    if z >= 0:
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        return 0.5 * (1 - math.erf(-z / math.sqrt(2)))


def chi_square_test(observed: List[int], expected: List[float]) -> Tuple[float, float]:
    """
    Perform Chi-square goodness of fit test.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        
    Returns:
        Tuple (chi_square_statistic, p_value_approximation)
        
    Time Complexity: O(k) where k is number of categories
    Space Complexity: O(1)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have same length")
    
    if any(e <= 0 for e in expected):
        raise ValueError("All expected frequencies must be positive")
    
    chi_square = sum((obs - exp) ** 2 / exp 
                    for obs, exp in zip(observed, expected))
    
    degrees_of_freedom = len(observed) - 1
    
    # Simple approximation for p-value (not exact)
    # For proper p-value, would need gamma function implementation
    if degrees_of_freedom == 1:
        p_value = 2 * (1 - normal_cdf(math.sqrt(chi_square)))
    else:
        # Very rough approximation
        p_value = math.exp(-chi_square / 2) if chi_square < 10 else 0.0
    
    return chi_square, p_value


def t_test_one_sample(data: List[float], mu0: float) -> Tuple[float, float]:
    """
    Perform one-sample t-test.
    
    Tests H0: mean = mu0 vs H1: mean ≠ mu0
    
    Args:
        data: Sample data
        mu0: Hypothesized population mean
        
    Returns:
        Tuple (t_statistic, p_value_approximation)
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for t-test")
    
    sample_mean = mean(data)
    sample_std = standard_deviation(data, population=False)
    n = len(data)
    
    t_stat = (sample_mean - mu0) / (sample_std / math.sqrt(n))
    
    # Approximate p-value using normal distribution (works for large n)
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    
    return t_stat, p_value


def percentile(data: List[float], p: float) -> float:
    """
    Calculate percentile of data.
    
    Args:
        data: List of numerical values
        p: Percentile (0 to 100)
        
    Returns:
        Value at given percentile
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    if not data:
        raise ValueError("Cannot calculate percentile of empty list")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if p == 0:
        return sorted_data[0]
    if p == 100:
        return sorted_data[-1]
    
    # Linear interpolation method
    index = (p / 100) * (n - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)
    
    if lower_index == upper_index:
        return sorted_data[lower_index]
    
    # Interpolate
    weight = index - lower_index
    return (1 - weight) * sorted_data[lower_index] + weight * sorted_data[upper_index]


def quartiles(data: List[float]) -> Tuple[float, float, float]:
    """
    Calculate first, second (median), and third quartiles.
    
    Args:
        data: List of numerical values
        
    Returns:
        Tuple (Q1, Q2, Q3)
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    q1 = percentile(data, 25)
    q2 = percentile(data, 50)  # median
    q3 = percentile(data, 75)
    
    return q1, q2, q3


def interquartile_range(data: List[float]) -> float:
    """
    Calculate interquartile range (Q3 - Q1).
    
    Args:
        data: List of numerical values
        
    Returns:
        Interquartile range
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    q1, _, q3 = quartiles(data)
    return q3 - q1


def outliers_iqr(data: List[float], factor: float = 1.5) -> List[float]:
    """
    Detect outliers using IQR method.
    
    Args:
        data: List of numerical values
        factor: IQR factor (typically 1.5)
        
    Returns:
        List of outlier values
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    q1, _, q3 = quartiles(data)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return [x for x in data if x < lower_bound or x > upper_bound]


def z_score(data: List[float]) -> List[float]:
    """
    Calculate z-scores for all data points.
    
    Args:
        data: List of numerical values
        
    Returns:
        List of z-scores
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for z-score")
    
    data_mean = mean(data)
    data_std = standard_deviation(data, population=False)
    
    if data_std == 0:
        return [0.0] * len(data)  # All values are the same
    
    return [(x - data_mean) / data_std for x in data]


def confidence_interval_mean(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for population mean.
    
    Args:
        data: Sample data
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple (lower_bound, upper_bound)
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")
    
    sample_mean = mean(data)
    sample_std = standard_deviation(data, population=False)
    n = len(data)
    
    # Use normal approximation for large samples
    alpha = 1 - confidence_level
    z_critical = 1.96  # Approximate for 95% confidence
    
    if confidence_level == 0.95:
        z_critical = 1.96
    elif confidence_level == 0.99:
        z_critical = 2.576
    elif confidence_level == 0.90:
        z_critical = 1.645
    else:
        # Rough approximation
        z_critical = abs(normal_cdf(alpha/2)) * 2
    
    margin_error = z_critical * (sample_std / math.sqrt(n))
    
    return sample_mean - margin_error, sample_mean + margin_error
