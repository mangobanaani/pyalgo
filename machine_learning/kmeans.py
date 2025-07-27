"""
K-Means Clustering Implementation

Unsupervised clustering algorithm that partitions data into k clusters
by minimizing within-cluster sum of squares.
"""

import math
import random
from typing import List, Tuple, Optional, Union


class KMeans:
    """
    K-Means clustering algorithm.
    
    Partitions data into k clusters by iteratively updating cluster centers
    to minimize within-cluster sum of squared distances.
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++',
                 max_iter: int = 300, tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_: Optional[List[List[float]]] = None
        self.labels_: Optional[List[int]] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        
        # Set random seed
        if self.random_state is not None:
            random.seed(self.random_state)
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def _calculate_centroid(self, points: List[List[float]]) -> List[float]:
        """Calculate centroid of a set of points."""
        if not points:
            return [0.0] * len(points[0]) if points else []
        
        n_features = len(points[0])
        centroid = [0.0] * n_features
        
        for point in points:
            for i in range(n_features):
                centroid[i] += point[i]
        
        return [coord / len(points) for coord in centroid]
    
    def _init_centroids_random(self, X: List[List[float]]) -> List[List[float]]:
        """Initialize centroids randomly."""
        n_samples, n_features = len(X), len(X[0])
        centroids = []
        
        for _ in range(self.n_clusters):
            # Random point from data
            centroid = X[random.randint(0, n_samples - 1)].copy()
            centroids.append(centroid)
        
        return centroids
    
    def _init_centroids_plus_plus(self, X: List[List[float]]) -> List[List[float]]:
        """Initialize centroids using k-means++ method."""
        n_samples = len(X)
        centroids = []
        
        # Choose first centroid randomly
        first_centroid = X[random.randint(0, n_samples - 1)].copy()
        centroids.append(first_centroid)
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            distances = []
            
            # Calculate distance to nearest centroid for each point
            for point in X:
                min_dist = float('inf')
                for centroid in centroids:
                    dist = self._euclidean_distance(point, centroid)
                    min_dist = min(min_dist, dist)
                distances.append(min_dist ** 2)
            
            # Choose next centroid with probability proportional to squared distance
            total_dist = sum(distances)
            if total_dist == 0:
                # All points are identical, choose randomly
                next_centroid = X[random.randint(0, n_samples - 1)].copy()
            else:
                probabilities = [d / total_dist for d in distances]
                
                # Weighted random selection
                r = random.random()
                cumulative_prob = 0
                for i, prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if r <= cumulative_prob:
                        next_centroid = X[i].copy()
                        break
                else:
                    next_centroid = X[-1].copy()
            
            centroids.append(next_centroid)
        
        return centroids
    
    def _assign_clusters(self, X: List[List[float]], 
                        centroids: List[List[float]]) -> List[int]:
        """Assign each point to nearest centroid."""
        labels = []
        
        for point in X:
            min_dist = float('inf')
            closest_cluster = 0
            
            for i, centroid in enumerate(centroids):
                dist = self._euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = i
            
            labels.append(closest_cluster)
        
        return labels
    
    def _update_centroids(self, X: List[List[float]], 
                         labels: List[int]) -> List[List[float]]:
        """Update centroids based on current cluster assignments."""
        n_features = len(X[0])
        new_centroids = []
        
        for cluster_id in range(self.n_clusters):
            # Get all points in this cluster
            cluster_points = [X[i] for i, label in enumerate(labels) if label == cluster_id]
            
            if cluster_points:
                centroid = self._calculate_centroid(cluster_points)
            else:
                # Empty cluster, reinitialize randomly
                centroid = [random.uniform(
                    min(point[j] for point in X),
                    max(point[j] for point in X)
                ) for j in range(n_features)]
            
            new_centroids.append(centroid)
        
        return new_centroids
    
    def _calculate_inertia(self, X: List[List[float]], labels: List[int],
                          centroids: List[List[float]]) -> float:
        """Calculate within-cluster sum of squares."""
        inertia = 0.0
        
        for i, point in enumerate(X):
            cluster_id = labels[i]
            centroid = centroids[cluster_id]
            dist = self._euclidean_distance(point, centroid)
            inertia += dist ** 2
        
        return inertia
    
    def _centroids_converged(self, old_centroids: List[List[float]],
                           new_centroids: List[List[float]]) -> bool:
        """Check if centroids have converged."""
        for old_centroid, new_centroid in zip(old_centroids, new_centroids):
            dist = self._euclidean_distance(old_centroid, new_centroid)
            if dist > self.tol:
                return False
        return True
    
    def fit(self, X: List[List[float]]) -> 'KMeans':
        """
        Fit K-Means clustering to data.
        
        Time Complexity: O(n * k * i * d) where n=samples, k=clusters, i=iterations, d=features
        
        Args:
            X: Training data (list of feature vectors)
            
        Returns:
            Self for method chaining
        """
        if not X or not X[0]:
            raise ValueError("Empty input data")
        
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        
        if self.n_clusters > len(X):
            raise ValueError("Number of clusters cannot exceed number of samples")
        
        # Initialize centroids
        if self.init == 'k-means++':
            centroids = self._init_centroids_plus_plus(X)
        else:
            centroids = self._init_centroids_random(X)
        
        # Main k-means loop
        for iteration in range(self.max_iter):
            # Assign points to clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if self._centroids_converged(centroids, new_centroids):
                self.n_iter_ = iteration + 1
                break
            
            centroids = new_centroids
        else:
            self.n_iter_ = self.max_iter
        
        # Store results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels, centroids)
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data to predict
            
        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """Fit model and return cluster labels for training data."""
        return self.fit(X).labels_
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Transform data to cluster-distance space.
        
        Args:
            X: Data to transform
            
        Returns:
            Distances to each cluster center
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")
        
        transformed = []
        for point in X:
            distances = []
            for centroid in self.cluster_centers_:
                dist = self._euclidean_distance(point, centroid)
                distances.append(dist)
            transformed.append(distances)
        
        return transformed
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """Fit model and transform training data."""
        self.fit(X)
        return self.transform(X)
    
    def score(self, X: List[List[float]]) -> float:
        """Return negative inertia (higher is better)."""
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")
        
        labels = self.predict(X)
        inertia = self._calculate_inertia(X, labels, self.cluster_centers_)
        return -inertia


class KMeansPlus(KMeans):
    """K-Means with k-means++ initialization (alias for convenience)."""
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 tol: float = 1e-4, random_state: Optional[int] = None):
        super().__init__(n_clusters=n_clusters, init='k-means++',
                        max_iter=max_iter, tol=tol, random_state=random_state)


class MiniBatchKMeans:
    """
    Mini-batch K-Means clustering.
    
    More efficient variant of K-Means that uses mini-batches for updates.
    """
    
    def __init__(self, n_clusters: int = 8, batch_size: int = 100,
                 max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        Initialize Mini-batch K-Means.
        
        Args:
            n_clusters: Number of clusters
            batch_size: Size of mini-batches
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_: Optional[List[List[float]]] = None
        self.labels_: Optional[List[int]] = None
        self.n_iter_: int = 0
        
        if self.random_state is not None:
            random.seed(self.random_state)
    
    def _get_mini_batch(self, X: List[List[float]]) -> List[List[float]]:
        """Get random mini-batch from data."""
        n_samples = len(X)
        batch_size = min(self.batch_size, n_samples)
        
        indices = random.sample(range(n_samples), batch_size)
        return [X[i] for i in indices]
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def _update_centroid(self, centroid: List[float], point: List[float],
                        count: int) -> List[float]:
        """Update centroid using streaming average."""
        learning_rate = 1.0 / count
        return [c + learning_rate * (p - c) for c, p in zip(centroid, point)]
    
    def fit(self, X: List[List[float]]) -> 'MiniBatchKMeans':
        """
        Fit Mini-batch K-Means to data.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        if not X or not X[0]:
            raise ValueError("Empty input data")
        
        n_samples, n_features = len(X), len(X[0])
        
        # Initialize centroids randomly
        centroids = []
        for _ in range(self.n_clusters):
            centroid = [random.uniform(
                min(point[j] for point in X),
                max(point[j] for point in X)
            ) for j in range(n_features)]
            centroids.append(centroid)
        
        # Keep track of assignment counts
        counts = [1] * self.n_clusters
        
        # Mini-batch updates
        for iteration in range(self.max_iter):
            old_centroids = [c.copy() for c in centroids]
            
            # Get mini-batch
            batch = self._get_mini_batch(X)
            
            # Update centroids based on mini-batch
            for point in batch:
                # Find closest centroid
                min_dist = float('inf')
                closest_cluster = 0
                
                for i, centroid in enumerate(centroids):
                    dist = self._euclidean_distance(point, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = i
                
                # Update centroid
                counts[closest_cluster] += 1
                centroids[closest_cluster] = self._update_centroid(
                    centroids[closest_cluster], point, counts[closest_cluster])
            
            # Check convergence
            converged = True
            for old_c, new_c in zip(old_centroids, centroids):
                if self._euclidean_distance(old_c, new_c) > self.tol:
                    converged = False
                    break
            
            if converged:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        # Final assignment
        self.cluster_centers_ = centroids
        self.labels_ = self._assign_clusters(X, centroids)
        
        return self
    
    def _assign_clusters(self, X: List[List[float]], 
                        centroids: List[List[float]]) -> List[int]:
        """Assign points to nearest centroids."""
        labels = []
        
        for point in X:
            min_dist = float('inf')
            closest_cluster = 0
            
            for i, centroid in enumerate(centroids):
                dist = self._euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = i
            
            labels.append(closest_cluster)
        
        return labels
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster labels."""
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return self._assign_clusters(X, self.cluster_centers_)


class ElbowMethod:
    """
    Elbow method for finding optimal number of clusters.
    
    Fits K-Means for different values of k and analyzes inertia.
    """
    
    def __init__(self, k_range: Tuple[int, int] = (1, 10), 
                 random_state: Optional[int] = None):
        """
        Initialize elbow method.
        
        Args:
            k_range: Range of k values to test (min_k, max_k)
            random_state: Random seed
        """
        self.k_range = k_range
        self.random_state = random_state
        self.inertias_: List[float] = []
        self.k_values_: List[int] = []
    
    def fit(self, X: List[List[float]]) -> 'ElbowMethod':
        """
        Fit K-Means for different k values.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        self.inertias_ = []
        self.k_values_ = list(range(self.k_range[0], self.k_range[1] + 1))
        
        for k in self.k_values_:
            if k > len(X):
                break
            
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            self.inertias_.append(kmeans.inertia_)
        
        return self
    
    def find_elbow(self) -> int:
        """
        Find elbow point using rate of change.
        
        Returns:
            Optimal number of clusters
        """
        if len(self.inertias_) < 3:
            return self.k_values_[0] if self.k_values_ else 1
        
        # Calculate rate of change
        rates = []
        for i in range(1, len(self.inertias_)):
            rate = abs(self.inertias_[i] - self.inertias_[i-1])
            rates.append(rate)
        
        # Find point where rate of change decreases most
        max_decrease = 0
        elbow_idx = 0
        
        for i in range(1, len(rates)):
            decrease = rates[i-1] - rates[i]
            if decrease > max_decrease:
                max_decrease = decrease
                elbow_idx = i
        
        return self.k_values_[elbow_idx + 1]  # +1 because rates has one less element
