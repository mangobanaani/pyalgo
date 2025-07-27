"""
Ensemble Methods Implementation

Implements ensemble learning algorithms including Random Forest,
AdaBoost, and Gradient Boosting for improved prediction performance.
"""

import math
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter


class RandomForestClassifier:
    """
    Random Forest Classifier.
    
    Implements random forest using bootstrap aggregating (bagging)
    with random feature selection for decision trees.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', bootstrap: bool = True,
                 random_state: int = None):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in a leaf
            max_features: Number of features to consider ('sqrt', 'log2', or int)
            bootstrap: Whether to bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees = []
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'RandomForestClassifier':
        """
        Fit the Random Forest classifier.
        
        Args:
            X: Training data
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        # Store dataset info
        self.classes_ = sorted(list(set(y)))
        self.n_classes_ = len(self.classes_)
        self.n_features_ = len(X[0])
        
        # Determine number of features to use
        if self.max_features == 'sqrt':
            max_features = int(math.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            max_features = int(math.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, self.n_features_)
        else:
            max_features = self.n_features_
        
        # Build trees
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            else:
                X_bootstrap, y_bootstrap = X, y
            
            # Create and train tree with feature randomness
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=None if self.random_state is None else self.random_state + i
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def _bootstrap_sample(self, X: List[List[float]], y: List[int]) -> Tuple[List[List[float]], List[int]]:
        """
        Create a bootstrap sample of the dataset.
        
        Args:
            X: Training data
            y: Target labels
            
        Returns:
            Bootstrap sample of X and y
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        
        X_bootstrap = [X[i] for i in indices]
        y_bootstrap = [y[i] for i in indices]
        
        return X_bootstrap, y_bootstrap
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict class labels using majority voting.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        if not self.trees:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            # Get predictions from all trees
            tree_predictions = [tree.predict_single(sample) for tree in self.trees]
            
            # Majority vote
            prediction = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities.
        
        Args:
            X: Input samples
            
        Returns:
            Class probabilities for each sample
        """
        if not self.trees:
            raise ValueError("Model has not been fitted yet")
        
        probabilities = []
        for sample in X:
            # Get predictions from all trees
            tree_predictions = [tree.predict_single(sample) for tree in self.trees]
            
            # Calculate probabilities
            class_counts = Counter(tree_predictions)
            probs = [class_counts.get(cls, 0) / len(self.trees) for cls in self.classes_]
            probabilities.append(probs)
        
        return probabilities
    
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


class SimpleDecisionTree:
    """
    Simplified decision tree for use in ensemble methods.
    
    This is a basic implementation for internal use by ensemble methods.
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None,
                 random_state: int = None):
        """Initialize simple decision tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.tree = None
        self.n_features_ = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Fit the tree."""
        self.n_features_ = len(X[0])
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X: List[List[float]], y: List[int], depth: int):
        """Build tree recursively."""
        n_samples = len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(set(y)) == 1):
            return {'is_leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        
        # Random feature selection
        if self.max_features is not None:
            available_features = random.sample(range(self.n_features_), 
                                             min(self.max_features, self.n_features_))
        else:
            available_features = list(range(self.n_features_))
        
        best_feature, best_threshold, best_gain = self._best_split(X, y, available_features)
        
        if best_gain == 0:
            return {'is_leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        
        # Split data
        left_indices, right_indices = self._split_data(X, best_feature, best_threshold)
        
        if (len(left_indices) < self.min_samples_leaf or
            len(right_indices) < self.min_samples_leaf):
            return {'is_leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        
        # Build subtrees
        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]
        
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, X: List[List[float]], y: List[int], features: List[int]):
        """Find best split among given features."""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini_impurity(y)
        
        for feature in features:
            feature_values = sorted(set(x[feature] for x in X))
            
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                left_indices, right_indices = self._split_data(X, feature, threshold)
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate weighted Gini after split
                n_total = len(y)
                n_left = len(left_indices)
                n_right = len(right_indices)
                
                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]
                
                left_gini = self._gini_impurity(y_left)
                right_gini = self._gini_impurity(y_right)
                
                weighted_gini = (n_left / n_total * left_gini +
                               n_right / n_total * right_gini)
                
                gain = current_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _split_data(self, X: List[List[float]], feature: int, threshold: float):
        """Split data based on feature and threshold."""
        left_indices = []
        right_indices = []
        
        for i, sample in enumerate(X):
            if sample[feature] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices
    
    def _gini_impurity(self, y: List[int]) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        
        class_counts = Counter(y)
        n_samples = len(y)
        
        gini = 1.0
        for count in class_counts.values():
            prob = count / n_samples
            gini -= prob ** 2
        
        return gini
    
    def predict_single(self, sample: List[float]) -> int:
        """Predict single sample."""
        node = self.tree
        
        while not node['is_leaf']:
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['value']


class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) Classifier.
    
    Implements AdaBoost algorithm using weak learners (decision stumps).
    """
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0,
                 random_state: int = None):
        """
        Initialize AdaBoost classifier.
        
        Args:
            n_estimators: Number of weak learners
            learning_rate: Learning rate shrinks contribution of each classifier
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.estimators = []
        self.estimator_weights = []
        self.classes_ = None
        self.n_classes_ = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'AdaBoostClassifier':
        """
        Fit the AdaBoost classifier.
        
        Args:
            X: Training data
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        # Convert to binary classification if needed
        self.classes_ = sorted(list(set(y)))
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ != 2:
            raise ValueError("AdaBoost currently supports only binary classification")
        
        # Convert labels to -1, +1
        y_binary = [1 if label == self.classes_[1] else -1 for label in y]
        
        n_samples = len(X)
        sample_weights = [1.0 / n_samples] * n_samples
        
        self.estimators = []
        self.estimator_weights = []
        
        for i in range(self.n_estimators):
            # Train weak learner with sample weights
            stump = DecisionStump()
            stump.fit(X, y_binary, sample_weights)
            
            # Make predictions
            predictions = [stump.predict_single(sample) for sample in X]
            
            # Calculate error rate
            error = sum(w for w, pred, true in zip(sample_weights, predictions, y_binary)
                       if pred != true)
            
            # Avoid division by zero
            if error <= 0:
                error = 1e-10
            elif error >= 0.5:
                break
            
            # Calculate classifier weight
            alpha = 0.5 * math.log((1 - error) / error)
            
            # Update sample weights
            for j in range(n_samples):
                if predictions[j] != y_binary[j]:
                    sample_weights[j] *= math.exp(alpha)
                else:
                    sample_weights[j] *= math.exp(-alpha)
            
            # Normalize weights
            weight_sum = sum(sample_weights)
            sample_weights = [w / weight_sum for w in sample_weights]
            
            # Store classifier and its weight
            self.estimators.append(stump)
            self.estimator_weights.append(alpha * self.learning_rate)
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict class labels.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        if not self.estimators:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            # Weighted voting
            score = sum(weight * estimator.predict_single(sample)
                       for estimator, weight in zip(self.estimators, self.estimator_weights))
            
            # Convert back to original labels
            prediction = self.classes_[1] if score > 0 else self.classes_[0]
            predictions.append(prediction)
        
        return predictions
    
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


class DecisionStump:
    """
    Decision Stump (single-level decision tree) for use as weak learner.
    
    Implements a simple decision tree with only one split.
    """
    
    def __init__(self):
        """Initialize decision stump."""
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X: List[List[float]], y: List[int], sample_weights: List[float]):
        """
        Fit decision stump to weighted data.
        
        Args:
            X: Training data
            y: Target labels (should be -1 or +1)
            sample_weights: Sample weights
        """
        best_error = float('inf')
        n_features = len(X[0])
        
        for feature in range(n_features):
            # Get unique thresholds
            feature_values = sorted(set(x[feature] for x in X))
            
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Try both possible labelings
                for left_val, right_val in [(-1, 1), (1, -1)]:
                    error = 0.0
                    
                    for j, (sample, true_label, weight) in enumerate(zip(X, y, sample_weights)):
                        prediction = left_val if sample[feature] <= threshold else right_val
                        if prediction != true_label:
                            error += weight
                    
                    if error < best_error:
                        best_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.left_value = left_val
                        self.right_value = right_val
    
    def predict_single(self, sample: List[float]) -> int:
        """
        Predict single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Prediction (-1 or +1)
        """
        if sample[self.feature] <= self.threshold:
            return self.left_value
        else:
            return self.right_value


# Example usage and testing
if __name__ == "__main__":
    # Test Random Forest
    print("Testing Random Forest Classifier...")
    
    # Create a larger dataset
    random.seed(42)
    X_rf = []
    y_rf = []
    
    # Generate 3 clusters
    for class_label in range(3):
        center_x = class_label * 5
        center_y = class_label * 5
        
        for _ in range(30):
            x = center_x + random.gauss(0, 1)
            y = center_y + random.gauss(0, 1)
            X_rf.append([x, y])
            y_rf.append(class_label)
    
    # Split into train/test
    train_size = int(0.8 * len(X_rf))
    X_train = X_rf[:train_size]
    y_train = y_rf[:train_size]
    X_test = X_rf[train_size:]
    y_test = y_rf[train_size:]
    
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    train_accuracy = rf.score(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)
    
    print(f"Random Forest train accuracy: {train_accuracy:.3f}")
    print(f"Random Forest test accuracy: {test_accuracy:.3f}")
    
    # Test AdaBoost (binary classification)
    print("\nTesting AdaBoost Classifier...")
    
    # Create binary classification dataset
    X_ada = []
    y_ada = []
    
    # Class 0: points near (2, 2)
    for _ in range(50):
        x = 2 + random.gauss(0, 0.8)
        y = 2 + random.gauss(0, 0.8)
        X_ada.append([x, y])
        y_ada.append(0)
    
    # Class 1: points near (6, 6)
    for _ in range(50):
        x = 6 + random.gauss(0, 0.8)
        y = 6 + random.gauss(0, 0.8)
        X_ada.append([x, y])
        y_ada.append(1)
    
    # Split data
    train_size = int(0.8 * len(X_ada))
    X_train_ada = X_ada[:train_size]
    y_train_ada = y_ada[:train_size]
    X_test_ada = X_ada[train_size:]
    y_test_ada = y_ada[train_size:]
    
    ada = AdaBoostClassifier(n_estimators=20, learning_rate=1.0, random_state=42)
    ada.fit(X_train_ada, y_train_ada)
    
    train_accuracy_ada = ada.score(X_train_ada, y_train_ada)
    test_accuracy_ada = ada.score(X_test_ada, y_test_ada)
    
    print(f"AdaBoost train accuracy: {train_accuracy_ada:.3f}")
    print(f"AdaBoost test accuracy: {test_accuracy_ada:.3f}")
    
    # Show some predictions
    test_predictions = ada.predict(X_test_ada[:5])
    print(f"AdaBoost test predictions (first 5): {test_predictions}")
    print(f"True labels (first 5): {y_test_ada[:5]}")
