"""
Decision Tree Implementation

Implements decision trees for classification and regression using
various splitting criteria and pruning techniques.
"""

import math
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter


class TreeNode:
    """Node in a decision tree."""
    
    def __init__(self, feature: int = None, threshold: float = None,
                 left: 'TreeNode' = None, right: 'TreeNode' = None,
                 value: Any = None, samples: int = 0):
        """
        Initialize tree node.
        
        Args:
            feature: Feature index for splitting
            threshold: Threshold value for splitting
            left: Left child node
            right: Right child node
            value: Prediction value for leaf nodes
            samples: Number of samples in this node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.is_leaf = value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier.
    
    Implements a decision tree for classification using entropy
    or Gini impurity as splitting criteria.
    """
    
    def __init__(self, criterion: str = 'gini', max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = None):
        """
        Initialize decision tree classifier.
        
        Args:
            criterion: Splitting criterion ('gini' or 'entropy')
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in a leaf
            random_state: Random seed for reproducibility
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'DecisionTreeClassifier':
        """
        Fit the decision tree classifier.
        
        Args:
            X: Training data
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Store dataset info
        self.classes_ = sorted(list(set(y)))
        self.n_classes_ = len(self.classes_)
        self.n_features_ = len(X[0])
        
        # Build the tree
        self.tree = self._build_tree(X, y, depth=0)
        
        return self
    
    def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> TreeNode:
        """
        Recursively build the decision tree.
        
        Args:
            X: Training data
            y: Target labels
            depth: Current depth
            
        Returns:
            Root node of subtree
        """
        n_samples = len(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(set(y)) == 1):
            # Create leaf node
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            # No improvement possible, create leaf
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Split the data
        left_indices, right_indices = self._split_data(X, best_feature, best_threshold)
        
        # Check minimum samples per leaf
        if (len(left_indices) < self.min_samples_leaf or
            len(right_indices) < self.min_samples_leaf):
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Create child datasets
        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]
        
        # Recursively build subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return TreeNode(feature=best_feature, threshold=best_threshold,
                       left=left_child, right=right_child, samples=n_samples)
    
    def _best_split(self, X: List[List[float]], y: List[int]) -> Tuple[int, float, float]:
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Training data
            y: Target labels
            
        Returns:
            Tuple of (best_feature, best_threshold, best_gain)
        """
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        # Try each feature
        for feature in range(self.n_features_):
            # Get unique values for this feature
            feature_values = sorted(set(x[feature] for x in X))
            
            # Try each potential threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_indices, right_indices = self._split_data(X, feature, threshold)
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate weighted impurity after split
                n_total = len(y)
                n_left = len(left_indices)
                n_right = len(right_indices)
                
                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]
                
                left_impurity = self._calculate_impurity(y_left)
                right_impurity = self._calculate_impurity(y_right)
                
                weighted_impurity = (n_left / n_total * left_impurity +
                                   n_right / n_total * right_impurity)
                
                # Calculate information gain
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _split_data(self, X: List[List[float]], feature: int, threshold: float) -> Tuple[List[int], List[int]]:
        """
        Split data based on feature and threshold.
        
        Args:
            X: Training data
            feature: Feature index
            threshold: Threshold value
            
        Returns:
            Tuple of (left_indices, right_indices)
        """
        left_indices = []
        right_indices = []
        
        for i, sample in enumerate(X):
            if sample[feature] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices
    
    def _calculate_impurity(self, y: List[int]) -> float:
        """
        Calculate impurity using specified criterion.
        
        Args:
            y: Target labels
            
        Returns:
            Impurity value
        """
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _gini_impurity(self, y: List[int]) -> float:
        """
        Calculate Gini impurity.
        
        Args:
            y: Target labels
            
        Returns:
            Gini impurity
        """
        if len(y) == 0:
            return 0
        
        class_counts = Counter(y)
        n_samples = len(y)
        
        gini = 1.0
        for count in class_counts.values():
            prob = count / n_samples
            gini -= prob ** 2
        
        return gini
    
    def _entropy(self, y: List[int]) -> float:
        """
        Calculate entropy.
        
        Args:
            y: Target labels
            
        Returns:
            Entropy value
        """
        if len(y) == 0:
            return 0
        
        class_counts = Counter(y)
        n_samples = len(y)
        
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                prob = count / n_samples
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _majority_class(self, y: List[int]) -> int:
        """
        Find the majority class in a list of labels.
        
        Args:
            y: Target labels
            
        Returns:
            Majority class
        """
        if not y:
            return self.classes_[0]
        
        class_counts = Counter(y)
        return class_counts.most_common(1)[0][0]
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict class labels for samples.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        if self.tree is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            prediction = self._predict_single(sample)
            predictions.append(prediction)
        
        return predictions
    
    def _predict_single(self, sample: List[float]) -> int:
        """
        Predict class label for a single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Predicted class label
        """
        node = self.tree
        
        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def score(self, X: List[List[float]], y: List[int]) -> float:
        """
        Return accuracy score.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor.
    
    Implements a decision tree for regression using mean squared error
    as the splitting criterion.
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: int = None):
        """
        Initialize decision tree regressor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in a leaf
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree = None
        self.n_features_ = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'DecisionTreeRegressor':
        """
        Fit the decision tree regressor.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        self.n_features_ = len(X[0])
        
        # Build the tree
        self.tree = self._build_tree(X, y, depth=0)
        
        return self
    
    def _build_tree(self, X: List[List[float]], y: List[float], depth: int) -> TreeNode:
        """
        Recursively build the decision tree.
        
        Args:
            X: Training data
            y: Target values
            depth: Current depth
            
        Returns:
            Root node of subtree
        """
        n_samples = len(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self._variance(y) == 0):
            # Create leaf node
            leaf_value = sum(y) / len(y)  # Mean value
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            # No improvement possible, create leaf
            leaf_value = sum(y) / len(y)
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Split the data
        left_indices, right_indices = self._split_data(X, best_feature, best_threshold)
        
        # Check minimum samples per leaf
        if (len(left_indices) < self.min_samples_leaf or
            len(right_indices) < self.min_samples_leaf):
            leaf_value = sum(y) / len(y)
            return TreeNode(value=leaf_value, samples=n_samples)
        
        # Create child datasets
        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]
        
        # Recursively build subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return TreeNode(feature=best_feature, threshold=best_threshold,
                       left=left_child, right=right_child, samples=n_samples)
    
    def _best_split(self, X: List[List[float]], y: List[float]) -> Tuple[int, float, float]:
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Tuple of (best_feature, best_threshold, best_gain)
        """
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        current_variance = self._variance(y)
        
        # Try each feature
        for feature in range(self.n_features_):
            # Get unique values for this feature
            feature_values = sorted(set(x[feature] for x in X))
            
            # Try each potential threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_indices, right_indices = self._split_data(X, feature, threshold)
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate weighted variance after split
                n_total = len(y)
                n_left = len(left_indices)
                n_right = len(right_indices)
                
                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]
                
                left_variance = self._variance(y_left)
                right_variance = self._variance(y_right)
                
                weighted_variance = (n_left / n_total * left_variance +
                                   n_right / n_total * right_variance)
                
                # Calculate variance reduction (gain)
                gain = current_variance - weighted_variance
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _split_data(self, X: List[List[float]], feature: int, threshold: float) -> Tuple[List[int], List[int]]:
        """
        Split data based on feature and threshold.
        
        Args:
            X: Training data
            feature: Feature index
            threshold: Threshold value
            
        Returns:
            Tuple of (left_indices, right_indices)
        """
        left_indices = []
        right_indices = []
        
        for i, sample in enumerate(X):
            if sample[feature] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices
    
    def _variance(self, y: List[float]) -> float:
        """
        Calculate variance of target values.
        
        Args:
            y: Target values
            
        Returns:
            Variance
        """
        if len(y) == 0:
            return 0
        
        mean = sum(y) / len(y)
        variance = sum((val - mean) ** 2 for val in y) / len(y)
        return variance
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict target values for samples.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted target values
        """
        if self.tree is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            prediction = self._predict_single(sample)
            predictions.append(prediction)
        
        return predictions
    
    def _predict_single(self, sample: List[float]) -> float:
        """
        Predict target value for a single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Predicted target value
        """
        node = self.tree
        
        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def score(self, X: List[List[float]], y: List[float]) -> float:
        """
        Return R² score.
        
        Args:
            X: Test data
            y: True values
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        
        # Calculate R² score
        y_mean = sum(y) / len(y)
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y, predictions))
        ss_tot = sum((true - y_mean) ** 2 for true in y)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


# Example usage and testing
if __name__ == "__main__":
    # Test classification
    print("Testing Decision Tree Classifier...")
    
    # Simple 2D classification dataset
    X_class = [
        [1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
        [5, 5], [5, 6], [6, 5], [6, 6],  # Class 1
        [9, 1], [9, 2], [10, 1], [10, 2] # Class 2
    ]
    y_class = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X_class, y_class)
    
    predictions_class = clf.predict(X_class)
    accuracy = clf.score(X_class, y_class)
    
    print(f"Classification accuracy: {accuracy:.3f}")
    print(f"Predictions: {predictions_class}")
    print(f"True labels: {y_class}")
    
    # Test regression
    print("\nTesting Decision Tree Regressor...")
    
    # Simple regression dataset: y = x1 + x2
    X_reg = [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    y_reg = [2, 3, 3, 4, 6, 8, 10]
    
    reg = DecisionTreeRegressor(max_depth=3, random_state=42)
    reg.fit(X_reg, y_reg)
    
    predictions_reg = reg.predict(X_reg)
    r2_score = reg.score(X_reg, y_reg)
    
    print(f"Regression R² score: {r2_score:.3f}")
    print(f"Predictions: {[f'{p:.2f}' for p in predictions_reg]}")
    print(f"True values: {y_reg}")
    
    # Test on new data
    X_test = [[2.5, 2.5], [3.5, 1.5]]
    test_predictions = reg.predict(X_test)
    print(f"Test predictions: {[f'{p:.2f}' for p in test_predictions]}")
    print(f"Expected (approx): [5.0, 5.0]")
