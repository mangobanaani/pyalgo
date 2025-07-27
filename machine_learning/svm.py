"""
Support Vector Machine Implementation

Implements Support Vector Machines for classification and regression
using the Sequential Minimal Optimization (SMO) algorithm.
"""

import math
import random
from typing import List, Tuple, Optional, Callable


class SVM:
    """
    Support Vector Machine for binary classification.
    
    Implements SVM using the Sequential Minimal Optimization (SMO) algorithm
    with support for different kernel functions.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', 
                 gamma: float = 'scale', degree: int = 3,
                 tolerance: float = 1e-3, max_iter: int = 1000,
                 random_state: int = None):
        """
        Initialize SVM classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient (for rbf, poly, sigmoid)
            degree: Degree for polynomial kernel
            tolerance: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Model parameters
        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def _kernel_function(self, x1: List[float], x2: List[float]) -> float:
        """
        Compute kernel function between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        if self.kernel == 'linear':
            return sum(a * b for a, b in zip(x1, x2))
        
        elif self.kernel == 'poly':
            dot_product = sum(a * b for a, b in zip(x1, x2))
            return (dot_product + 1) ** self.degree
        
        elif self.kernel == 'rbf':
            # Compute gamma if set to 'scale'
            if self.gamma == 'scale':
                gamma_val = 1.0 / len(x1)
            else:
                gamma_val = self.gamma
            
            squared_distance = sum((a - b) ** 2 for a, b in zip(x1, x2))
            return math.exp(-gamma_val * squared_distance)
        
        elif self.kernel == 'sigmoid':
            # Compute gamma if set to 'scale'
            if self.gamma == 'scale':
                gamma_val = 1.0 / len(x1)
            else:
                gamma_val = self.gamma
            
            dot_product = sum(a * b for a, b in zip(x1, x2))
            return math.tanh(gamma_val * dot_product + 1)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_error(self, i: int) -> float:
        """
        Compute prediction error for sample i.
        
        Args:
            i: Sample index
            
        Returns:
            Prediction error
        """
        prediction = self._predict_value(self.X_train[i])
        return prediction - self.y_train[i]
    
    def _predict_value(self, x: List[float]) -> float:
        """
        Compute SVM decision function value.
        
        Args:
            x: Input vector
            
        Returns:
            Decision function value
        """
        result = 0.0
        for i in range(len(self.X_train)):
            if self.alphas[i] > 0:
                result += self.alphas[i] * self.y_train[i] * self._kernel_function(self.X_train[i], x)
        
        return result + self.b
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'SVM':
        """
        Fit the SVM model using SMO algorithm.
        
        Args:
            X: Training data
            y: Target labels (should be -1 or +1)
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Convert labels to -1, +1 if needed
        unique_labels = sorted(list(set(y)))
        if len(unique_labels) != 2:
            raise ValueError("SVM supports only binary classification")
        
        self.classes_ = unique_labels
        y_binary = [1 if label == unique_labels[1] else -1 for label in y]
        
        self.X_train = X
        self.y_train = y_binary
        n_samples = len(X)
        
        # Initialize alphas
        self.alphas = [0.0] * n_samples
        self.b = 0.0
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        
        for iteration in range(self.max_iter):
            num_changed = 0
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Examine non-bound samples (0 < alpha < C)
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            if examine_all and num_changed == 0:
                break
        
        # Extract support vectors
        self._extract_support_vectors()
        
        return self
    
    def _examine_example(self, i1: int) -> int:
        """
        Examine example i1 and try to find a second example to optimize.
        
        Args:
            i1: Index of first example
            
        Returns:
            1 if optimization occurred, 0 otherwise
        """
        y1 = self.y_train[i1]
        alpha1 = self.alphas[i1]
        E1 = self._compute_error(i1)
        r1 = E1 * y1
        
        # Check KKT conditions
        if ((r1 < -self.tolerance and alpha1 < self.C) or
            (r1 > self.tolerance and alpha1 > 0)):
            
            # Try to find second example
            # First, look for examples with maximum |E1 - E2|
            i2 = self._select_second_example(i1, E1)
            
            if i2 is not None and self._take_step(i1, i2):
                return 1
            
            # Loop over all non-zero and non-C alphas
            non_bound_indices = [i for i in range(len(self.alphas))
                               if 0 < self.alphas[i] < self.C]
            random.shuffle(non_bound_indices)
            
            for i2 in non_bound_indices:
                if self._take_step(i1, i2):
                    return 1
            
            # Loop over all examples
            all_indices = list(range(len(self.alphas)))
            random.shuffle(all_indices)
            
            for i2 in all_indices:
                if self._take_step(i1, i2):
                    return 1
        
        return 0
    
    def _select_second_example(self, i1: int, E1: float) -> Optional[int]:
        """
        Select second example for optimization using maximum |E1 - E2|.
        
        Args:
            i1: Index of first example
            E1: Error for first example
            
        Returns:
            Index of second example or None
        """
        best_i2 = None
        best_delta = 0
        
        for i2 in range(len(self.X_train)):
            if i2 == i1:
                continue
            
            E2 = self._compute_error(i2)
            delta = abs(E1 - E2)
            
            if delta > best_delta:
                best_delta = delta
                best_i2 = i2
        
        return best_i2 if best_delta > 0 else None
    
    def _take_step(self, i1: int, i2: int) -> bool:
        """
        Optimize alphas for examples i1 and i2.
        
        Args:
            i1: Index of first example
            i2: Index of second example
            
        Returns:
            True if optimization occurred
        """
        if i1 == i2:
            return False
        
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.y_train[i1]
        y2 = self.y_train[i2]
        
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        
        s = y1 * y2
        
        # Compute bounds L and H
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        
        if L == H:
            return False
        
        # Compute eta (second derivative of objective function)
        k11 = self._kernel_function(self.X_train[i1], self.X_train[i1])
        k12 = self._kernel_function(self.X_train[i1], self.X_train[i2])
        k22 = self._kernel_function(self.X_train[i2], self.X_train[i2])
        
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            # Compute new alpha2
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            
            # Clip to bounds
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # eta <= 0, compute objective function at endpoints
            return False  # Simplified: skip this case
        
        # Check if change is significant
        if abs(alpha2_new - alpha2) < 1e-5:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
        
        # Update threshold b
        b1 = (self.b - E1 - y1 * (alpha1_new - alpha1) * k11 - 
              y2 * (alpha2_new - alpha2) * k12)
        
        b2 = (self.b - E2 - y1 * (alpha1_new - alpha1) * k12 - 
              y2 * (alpha2_new - alpha2) * k22)
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new
        
        return True
    
    def _extract_support_vectors(self):
        """Extract support vectors from training data."""
        support_indices = [i for i, alpha in enumerate(self.alphas) if alpha > 1e-5]
        
        self.support_vectors = [self.X_train[i] for i in support_indices]
        self.support_vector_labels = [self.y_train[i] for i in support_indices]
        self.support_vector_alphas = [self.alphas[i] for i in support_indices]
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict class labels.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            prediction_value = self._predict_value(sample)
            prediction = self.classes_[1] if prediction_value > 0 else self.classes_[0]
            predictions.append(prediction)
        
        return predictions
    
    def decision_function(self, X: List[List[float]]) -> List[float]:
        """
        Compute decision function values.
        
        Args:
            X: Input samples
            
        Returns:
            Decision function values
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet")
        
        return [self._predict_value(sample) for sample in X]
    
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


class SVR:
    """
    Support Vector Regression.
    
    Implements epsilon-SVR for regression tasks.
    """
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, 
                 kernel: str = 'rbf', gamma: float = 'scale',
                 degree: int = 3, tolerance: float = 1e-3,
                 max_iter: int = 1000):
        """
        Initialize SVR.
        
        Args:
            C: Regularization parameter
            epsilon: Epsilon for epsilon-insensitive loss
            kernel: Kernel function
            gamma: Kernel coefficient
            degree: Degree for polynomial kernel
            tolerance: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
        """
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tolerance = tolerance
        self.max_iter = max_iter
        
        # Model parameters (simplified implementation)
        self.support_vectors = None
        self.support_vector_alphas = None
        self.support_vector_targets = None
        self.b = 0.0
    
    def _kernel_function(self, x1: List[float], x2: List[float]) -> float:
        """Compute kernel function (same as SVM)."""
        if self.kernel == 'linear':
            return sum(a * b for a, b in zip(x1, x2))
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma_val = 1.0 / len(x1)
            else:
                gamma_val = self.gamma
            squared_distance = sum((a - b) ** 2 for a, b in zip(x1, x2))
            return math.exp(-gamma_val * squared_distance)
        else:
            raise ValueError(f"Kernel {self.kernel} not implemented for SVR")
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'SVR':
        """
        Fit SVR model (simplified implementation).
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        # Simplified implementation: use all training points as support vectors
        # In practice, would use SMO algorithm similar to SVM
        self.support_vectors = X
        self.support_vector_targets = y
        
        # Initialize alphas (simplified)
        n_samples = len(X)
        self.support_vector_alphas = [0.1] * n_samples
        
        # Compute bias term (simplified)
        self.b = 0.0
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict target values.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted values
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for sample in X:
            prediction = 0.0
            for sv, alpha, target in zip(self.support_vectors, 
                                       self.support_vector_alphas,
                                       self.support_vector_targets):
                prediction += alpha * self._kernel_function(sample, sv)
            
            prediction += self.b
            predictions.append(prediction)
        
        return predictions
    
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
    # Test SVM classifier
    print("Testing SVM Classifier...")
    
    # Create linearly separable dataset
    random.seed(42)
    X_svm = []
    y_svm = []
    
    # Class 0: points in lower-left
    for _ in range(20):
        x = random.uniform(0, 2)
        y = random.uniform(0, 2)
        X_svm.append([x, y])
        y_svm.append(0)
    
    # Class 1: points in upper-right
    for _ in range(20):
        x = random.uniform(3, 5)
        y = random.uniform(3, 5)
        X_svm.append([x, y])
        y_svm.append(1)
    
    # Train SVM
    svm = SVM(C=1.0, kernel='linear', random_state=42)
    svm.fit(X_svm, y_svm)
    
    # Test predictions
    predictions = svm.predict(X_svm)
    accuracy = svm.score(X_svm, y_svm)
    
    print(f"SVM accuracy: {accuracy:.3f}")
    print(f"Number of support vectors: {len(svm.support_vectors)}")
    
    # Test with RBF kernel
    svm_rbf = SVM(C=1.0, kernel='rbf', gamma=0.1, random_state=42)
    svm_rbf.fit(X_svm, y_svm)
    
    accuracy_rbf = svm_rbf.score(X_svm, y_svm)
    print(f"SVM (RBF) accuracy: {accuracy_rbf:.3f}")
    
    # Test SVR
    print("\nTesting SVR...")
    
    # Create regression dataset
    X_reg = [[i] for i in range(10)]
    y_reg = [2 * i + 1 + random.gauss(0, 0.1) for i in range(10)]
    
    svr = SVR(C=1.0, epsilon=0.1, kernel='linear')
    svr.fit(X_reg, y_reg)
    
    predictions_reg = svr.predict(X_reg)
    r2_score = svr.score(X_reg, y_reg)
    
    print(f"SVR R² score: {r2_score:.3f}")
    print(f"Sample predictions: {[f'{p:.2f}' for p in predictions_reg[:5]]}")
    print(f"True values: {[f'{v:.2f}' for v in y_reg[:5]]}")
