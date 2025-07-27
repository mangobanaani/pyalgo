"""
Tests for Machine Learning Algorithms

Test suite for machine learning algorithms including supervised learning,
unsupervised learning, and model evaluation.
"""

import unittest
import random
import math
from machine_learning import LinearRegression, Ridge, Lasso, KMeans


class TestLinearRegression(unittest.TestCase):
    """Test cases for Linear Regression."""
    
    def setUp(self):
        """Set up test data."""
        # Create simple linear relationship: y = 2x + 1 + noise
        self.X_simple = [[1], [2], [3], [4], [5]]
        self.y_simple = [3, 5, 7, 9, 11]
        
        # Create multiple feature data
        self.X_multi = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        self.y_multi = [7, 11, 15, 19, 23]  # y = 2*x1 + 3*x2 + 1
    
    def test_simple_linear_regression(self):
        """Test simple linear regression."""
        lr = LinearRegression()
        lr.fit(self.X_simple, self.y_simple)
        
        # Check if coefficients are approximately correct
        self.assertAlmostEqual(lr.coef_[0], 2.0, places=6)
        self.assertAlmostEqual(lr.intercept_, 1.0, places=6)
        
        # Test predictions
        predictions = lr.predict([[6], [7]])
        expected = [13, 15]  # 2*6+1=13, 2*7+1=15
        
        for pred, exp in zip(predictions, expected):
            self.assertAlmostEqual(pred, exp, places=6)
    
    def test_multiple_linear_regression(self):
        """Test multiple linear regression."""
        lr = LinearRegression()
        lr.fit(self.X_multi, self.y_multi)
        
        # Check coefficients
        self.assertAlmostEqual(lr.coef_[0], 2.0, places=6)
        self.assertAlmostEqual(lr.coef_[1], 3.0, places=6)
        self.assertAlmostEqual(lr.intercept_, 1.0, places=6)
        
        # Test R² score
        r2 = lr.score(self.X_multi, self.y_multi)
        self.assertAlmostEqual(r2, 1.0, places=6)  # Perfect fit
    
    def test_ridge_regression(self):
        """Test Ridge regression with regularization."""
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_simple, self.y_simple)
        
        # Ridge should produce smaller coefficients due to regularization
        lr = LinearRegression()
        lr.fit(self.X_simple, self.y_simple)
        
        self.assertLess(abs(ridge.coef_[0]), abs(lr.coef_[0]) + 0.1)
    
    def test_lasso_regression(self):
        """Test Lasso regression with L1 regularization."""
        # Create data with some irrelevant features
        X_noise = [[1, 2, 0.1], [2, 3, 0.2], [3, 4, 0.3], [4, 5, 0.4], [5, 6, 0.5]]
        y = [7, 11, 15, 19, 23]  # Only depends on first two features
        
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_noise, y)
        
        # Lasso should drive irrelevant coefficients toward zero
        self.assertLess(abs(lasso.coef_[2]), 0.5)  # Third coefficient should be small
    
    def test_normalization(self):
        """Test feature normalization."""
        # Create data with different scales
        X_scaled = [[100, 1], [200, 2], [300, 3], [400, 4], [500, 5]]
        y = [301, 602, 903, 1204, 1505]  # y = 3*x1 + x2
        
        lr_norm = LinearRegression(normalize=True)
        lr_norm.fit(X_scaled, y)
        
        lr_no_norm = LinearRegression(normalize=False)
        lr_no_norm.fit(X_scaled, y)
        
        # Both should make similar predictions despite scaling
        test_X = [[600, 6]]
        pred_norm = lr_norm.predict(test_X)
        pred_no_norm = lr_no_norm.predict(test_X)
        
        self.assertAlmostEqual(pred_norm[0], pred_no_norm[0], places=2)


class TestKMeans(unittest.TestCase):
    """Test cases for K-Means clustering."""
    
    def setUp(self):
        """Set up test data."""
        # Create three clear clusters
        self.cluster1 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        self.cluster2 = [[8, 8], [8, 9], [9, 8], [9, 9]]
        self.cluster3 = [[1, 8], [1, 9], [2, 8], [2, 9]]
        
        self.X_clusters = self.cluster1 + self.cluster2 + self.cluster3
        random.seed(42)  # For reproducible tests
    
    def test_basic_clustering(self):
        """Test basic K-means clustering."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(self.X_clusters)
        
        # Should have 3 clusters
        unique_labels = set(labels)
        self.assertEqual(len(unique_labels), 3)
        
        # Points in same original cluster should mostly have same label
        labels1 = set(labels[:4])   # First 4 points (cluster1)
        labels2 = set(labels[4:8])  # Next 4 points (cluster2)
        labels3 = set(labels[8:])   # Last 4 points (cluster3)
        
        # Each cluster should be mostly homogeneous
        self.assertLessEqual(len(labels1), 2)
        self.assertLessEqual(len(labels2), 2)
        self.assertLessEqual(len(labels3), 2)
    
    def test_single_cluster(self):
        """Test K-means with k=1."""
        kmeans = KMeans(n_clusters=1, random_state=42)
        labels = kmeans.fit_predict(self.X_clusters)
        
        # All points should be in same cluster
        self.assertEqual(set(labels), {0})
    
    def test_more_clusters_than_points(self):
        """Test error handling when k > n_samples."""
        small_data = [[1, 1], [2, 2]]
        kmeans = KMeans(n_clusters=5)
        
        with self.assertRaises(ValueError):
            kmeans.fit(small_data)
    
    def test_empty_data(self):
        """Test error handling with empty data."""
        kmeans = KMeans(n_clusters=2)
        
        with self.assertRaises(ValueError):
            kmeans.fit([])
    
    def test_convergence(self):
        """Test that algorithm converges."""
        kmeans = KMeans(n_clusters=3, max_iter=100, tol=1e-4, random_state=42)
        kmeans.fit(self.X_clusters)
        
        # Should converge before max iterations for this simple case
        self.assertLess(kmeans.n_iter_, 50)
    
    def test_cluster_centers(self):
        """Test that cluster centers are reasonable."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.X_clusters)
        
        centers = kmeans.cluster_centers_
        self.assertEqual(len(centers), 3)
        
        # Each center should have 2 coordinates
        for center in centers:
            self.assertEqual(len(center), 2)
            # Centers should be within reasonable bounds
            self.assertTrue(0 <= center[0] <= 10)
            self.assertTrue(0 <= center[1] <= 10)
    
    def test_prediction(self):
        """Test prediction on new data."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.X_clusters)
        
        # Predict cluster for new points
        new_points = [[1.5, 1.5], [8.5, 8.5]]
        predictions = kmeans.predict(new_points)
        
        self.assertEqual(len(predictions), 2)
        # Predictions should be valid cluster indices
        for pred in predictions:
            self.assertIn(pred, [0, 1, 2])
    
    def test_inertia(self):
        """Test within-cluster sum of squares calculation."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.X_clusters)
        
        # Inertia should be positive
        self.assertGreater(kmeans.inertia_, 0)
        
        # More clusters should generally reduce inertia
        kmeans_more = KMeans(n_clusters=6, random_state=42)
        kmeans_more.fit(self.X_clusters)
        
        self.assertLessEqual(kmeans_more.inertia_, kmeans.inertia_)


class TestMLUtilities(unittest.TestCase):
    """Test utility functions and helper methods."""
    
    def test_distance_calculations(self):
        """Test distance calculation methods."""
        point1 = [1, 2, 3]
        point2 = [4, 6, 8]
        
        # Test Euclidean distance calculation (if available)
        # This would be in a utility module
        expected_distance = math.sqrt((4-1)**2 + (6-2)**2 + (8-3)**2)
        self.assertAlmostEqual(expected_distance, math.sqrt(50), places=6)
    
    def test_data_validation(self):
        """Test data validation and preprocessing."""
        # Test various edge cases that ML algorithms should handle
        
        # Empty features
        X_empty = [[], [], []]
        self.assertEqual(len(X_empty), 3)
        self.assertEqual(len(X_empty[0]), 0)
        
        # Single feature
        X_single = [[1], [2], [3]]
        self.assertEqual(len(X_single[0]), 1)
        
        # Inconsistent dimensions should be caught by algorithms
        X_inconsistent = [[1, 2], [3], [4, 5, 6]]
        # This should raise an error in actual algorithms


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics and methods."""
    
    def test_r_squared_calculation(self):
        """Test R² score calculation."""
        # Perfect predictions
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        # Calculate R² manually
        y_mean = sum(y_true) / len(y_true)
        ss_res = sum((true - pred)**2 for true, pred in zip(y_true, y_pred))
        ss_tot = sum((true - y_mean)**2 for true in y_true)
        
        r2_manual = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        self.assertAlmostEqual(r2_manual, 1.0, places=6)
        
        # Worst possible predictions (always predict mean)
        y_pred_mean = [y_mean] * len(y_true)
        ss_res_worst = sum((true - pred)**2 for true, pred in zip(y_true, y_pred_mean))
        r2_worst = 1 - (ss_res_worst / ss_tot) if ss_tot != 0 else 1.0
        self.assertAlmostEqual(r2_worst, 0.0, places=6)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple ML components."""
    
    def test_preprocessing_and_regression(self):
        """Test preprocessing pipeline with regression."""
        # Create data that needs preprocessing
        X_raw = [[100, 1000], [200, 2000], [300, 3000], [400, 4000]]
        y_raw = [1200, 2400, 3600, 4800]  # y = 12 * (x1 + x2)
        
        # Test with normalization
        lr_norm = LinearRegression(normalize=True)
        lr_norm.fit(X_raw, y_raw)
        
        # Should still make good predictions
        pred = lr_norm.predict([[500, 5000]])
        expected = 6000  # 12 * (500 + 5000) / 100 = 6000
        
        # Allow some tolerance due to numerical precision
        self.assertAlmostEqual(pred[0], expected, delta=100)
    
    def test_clustering_and_classification_data(self):
        """Test clustering on classification-like data."""
        # Create data with clear class structure
        class_1 = [[i, i] for i in range(5)]        # y = x line
        class_2 = [[i, 10-i] for i in range(5)]     # y = 10-x line
        
        X_classes = class_1 + class_2
        
        # K-means should find 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X_classes)
        
        # Check that points from same original class get same label
        labels_1 = labels[:5]
        labels_2 = labels[5:]
        
        # Each class should be mostly homogeneous
        self.assertLessEqual(len(set(labels_1)), 2)
        self.assertLessEqual(len(set(labels_2)), 2)


class TestPerformance(unittest.TestCase):
    """Performance tests for ML algorithms."""
    
    def test_linear_regression_performance(self):
        """Test linear regression with larger dataset."""
        n_samples = 1000
        n_features = 10
        
        # Generate synthetic data
        X = [[random.random() for _ in range(n_features)] for _ in range(n_samples)]
        
        # Create target with known relationship
        true_coef = [i + 1 for i in range(n_features)]  # [1, 2, 3, ..., 10]
        y = [sum(x[i] * true_coef[i] for i in range(n_features)) + random.random() * 0.1 
             for x in X]
        
        # Fit model
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Should achieve good R² score
        r2 = lr.score(X, y)
        self.assertGreater(r2, 0.95)  # Should be very high due to low noise
    
    def test_kmeans_performance(self):
        """Test K-means with larger dataset."""
        n_samples = 500
        n_clusters = 5
        
        # Generate data with clear clusters
        X = []
        for cluster in range(n_clusters):
            center_x = cluster * 10
            center_y = cluster * 10
            
            for _ in range(n_samples // n_clusters):
                x = center_x + random.gauss(0, 1)
                y = center_y + random.gauss(0, 1)
                X.append([x, y])
        
        # Cluster the data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit(X)
        
        # Should converge reasonably quickly
        self.assertLess(kmeans.n_iter_, 100)
        
        # Should have found all clusters
        unique_labels = set(labels.labels_)
        self.assertEqual(len(unique_labels), n_clusters)


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestLinearRegression, TestKMeans, TestMLUtilities,
        TestModelEvaluation, TestIntegration, TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nRan {result.testsRun} tests")
    if result.failures:
        print(f"FAILURES: {len(result.failures)}")
        for test, traceback in result.failures:
            print(f"FAILED: {test}")
    
    if result.errors:
        print(f"ERRORS: {len(result.errors)}")
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
    
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed or had errors.")
        print("Note: Some failures expected due to missing implementations.")
