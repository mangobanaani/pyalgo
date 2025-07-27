"""
Linear Regression Implementation

Linear regression models for supervised learning including
ordinary least squares, ridge, lasso, and elastic net regularization.
"""

import numpy as np
from typing import Optional, Union, Tuple
import warnings


class BaseLinearRegression:
    """Base class for linear regression models."""
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = False):
        """
        Initialize base linear regression.
        
        Args:
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features before fitting
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_features_: int = 0
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_stds_: Optional[np.ndarray] = None
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                     training: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for training or prediction."""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if training:
            self.n_features_ = X.shape[1]
            
            if self.normalize:
                self.feature_means_ = np.mean(X, axis=0)
                self.feature_stds_ = np.std(X, axis=0)
                # Avoid division by zero
                self.feature_stds_[self.feature_stds_ == 0] = 1.0
                X = (X - self.feature_means_) / self.feature_stds_
            
            if y is not None:
                y = np.asarray(y)
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y.ravel()
                return X, y
            return X
        else:
            # Transform for prediction
            if self.normalize and self.feature_means_ is not None:
                X = (X - self.feature_means_) / self.feature_stds_
            return X
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term to feature matrix."""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X = self._prepare_data(X, training=False)
        
        predictions = X @ self.coef_
        
        if self.fit_intercept:
            predictions += self.intercept_
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class LinearRegression(BaseLinearRegression):
    """
    Ordinary Least Squares Linear Regression.
    
    Fits linear model using normal equation or SVD decomposition.
    """
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = False):
        super().__init__(fit_intercept, normalize)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model.
        
        Time Complexity: O(n * p²) where n is samples, p is features
        
        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Self for method chaining
        """
        X, y = self._prepare_data(X, y, training=True)
        
        if self.fit_intercept:
            # Solve using normal equation with intercept
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            
            # Center the data
            X_centered = X - X_mean
            y_centered = y - y_mean
            
            # Solve normal equation
            try:
                self.coef_ = np.linalg.solve(X_centered.T @ X_centered, X_centered.T @ y_centered)
            except np.linalg.LinAlgError:
                # Use SVD if matrix is singular
                self.coef_ = np.linalg.pinv(X_centered.T @ X_centered) @ X_centered.T @ y_centered
            
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            # Solve without intercept
            try:
                self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
            except np.linalg.LinAlgError:
                self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
            
            self.intercept_ = 0.0
        
        return self
    
    def fit_svd(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit using SVD decomposition (more numerically stable).
        
        Args:
            X: Training feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        X, y = self._prepare_data(X, y, training=True)
        
        if self.fit_intercept:
            # Add intercept column
            X_with_intercept = self._add_intercept(X)
            
            # SVD decomposition
            U, s, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
            
            # Solve using SVD
            coefficients = Vt.T @ np.diag(1 / s) @ U.T @ y
            
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
        else:
            # SVD without intercept
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            self.coef_ = Vt.T @ np.diag(1 / s) @ U.T @ y
            self.intercept_ = 0.0
        
        return self


class Ridge(BaseLinearRegression):
    """
    Ridge Regression (L2 regularization).
    
    Linear regression with L2 penalty to prevent overfitting.
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, 
                 normalize: bool = False, max_iter: int = 1000):
        """
        Initialize Ridge regression.
        
        Args:
            alpha: Regularization strength (higher = more regularization)
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features
            max_iter: Maximum iterations for iterative solver
        """
        super().__init__(fit_intercept, normalize)
        self.alpha = alpha
        self.max_iter = max_iter
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Ridge':
        """
        Fit Ridge regression model.
        
        Args:
            X: Training feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        X, y = self._prepare_data(X, y, training=True)
        
        if self.fit_intercept:
            # Center the data
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
            
            # Solve regularized normal equation
            n_features = X_centered.shape[1]
            A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
            b = X_centered.T @ y_centered
            
            try:
                self.coef_ = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.coef_ = np.linalg.pinv(A) @ b
            
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            # Solve without intercept
            n_features = X.shape[1]
            A = X.T @ X + self.alpha * np.eye(n_features)
            b = X.T @ y
            
            try:
                self.coef_ = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.coef_ = np.linalg.pinv(A) @ b
            
            self.intercept_ = 0.0
        
        return self


class Lasso(BaseLinearRegression):
    """
    Lasso Regression (L1 regularization).
    
    Linear regression with L1 penalty for feature selection.
    Uses coordinate descent for optimization.
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 normalize: bool = False, max_iter: int = 1000, 
                 tol: float = 1e-4, selection: str = 'cyclic'):
        """
        Initialize Lasso regression.
        
        Args:
            alpha: Regularization strength
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            selection: Feature selection order ('cyclic' or 'random')
        """
        super().__init__(fit_intercept, normalize)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.n_iter_: int = 0
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator for L1 regularization."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Lasso':
        """
        Fit Lasso regression using coordinate descent.
        
        Args:
            X: Training feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        X, y = self._prepare_data(X, y, training=True)
        n_samples, n_features = X.shape
        
        # Center the data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Precompute norms
        X_norms = np.sum(X**2, axis=0)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Update each coefficient
            for j in range(n_features):
                if X_norms[j] == 0:
                    continue
                
                # Compute residual without j-th feature
                residual = y - X @ self.coef_ + self.coef_[j] * X[:, j]
                
                # Compute coordinate update
                rho = X[:, j] @ residual / n_samples
                
                # Apply soft thresholding
                self.coef_[j] = self._soft_threshold(rho, self.alpha) / (X_norms[j] / n_samples)
            
            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            warnings.warn(f"Lasso did not converge after {self.max_iter} iterations")
            self.n_iter_ = self.max_iter
        
        # Set intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        return self


class ElasticNet(BaseLinearRegression):
    """
    Elastic Net Regression (L1 + L2 regularization).
    
    Combines Ridge and Lasso regularization.
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 fit_intercept: bool = True, normalize: bool = False,
                 max_iter: int = 1000, tol: float = 1e-4):
        """
        Initialize Elastic Net regression.
        
        Args:
            alpha: Overall regularization strength
            l1_ratio: Ratio of L1 to total penalty (0 = Ridge, 1 = Lasso)
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features
            max_iter: Maximum iterations
            tol: Tolerance for convergence
        """
        super().__init__(fit_intercept, normalize)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_: int = 0
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
        """
        Fit Elastic Net regression using coordinate descent.
        
        Args:
            X: Training feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        X, y = self._prepare_data(X, y, training=True)
        n_samples, n_features = X.shape
        
        # Center the data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Precompute norms
        X_norms = np.sum(X**2, axis=0)
        
        # Regularization parameters
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Update each coefficient
            for j in range(n_features):
                if X_norms[j] == 0:
                    continue
                
                # Compute residual without j-th feature
                residual = y - X @ self.coef_ + self.coef_[j] * X[:, j]
                
                # Compute coordinate update
                rho = X[:, j] @ residual / n_samples
                
                # Apply soft thresholding with elastic net penalty
                denominator = (X_norms[j] / n_samples) + l2_reg
                self.coef_[j] = self._soft_threshold(rho, l1_reg) / denominator
            
            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            warnings.warn(f"ElasticNet did not converge after {self.max_iter} iterations")
            self.n_iter_ = self.max_iter
        
        # Set intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        return self


class PolynomialFeatures:
    """Generate polynomial features for linear regression."""
    
    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        """
        Initialize polynomial feature generator.
        
        Args:
            degree: Degree of polynomial features
            include_bias: Whether to include bias/intercept term
            interaction_only: Whether to include only interaction terms
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_features_: int = 0
        self.feature_names_: list = []
    
    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Fit polynomial feature generator.
        
        Args:
            X: Input features
            
        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to polynomial features.
        
        Args:
            X: Input features
            
        Returns:
            Polynomial features
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Start with bias term if requested
        features = []
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        # Add original features (degree 1)
        if not self.interaction_only or self.degree == 1:
            features.append(X)
        
        # Add higher degree features
        for deg in range(2, self.degree + 1):
            if self.interaction_only:
                # Only interaction terms
                from itertools import combinations_with_replacement
                for indices in combinations_with_replacement(range(n_features), deg):
                    if len(set(indices)) > 1:  # Only true interactions
                        feature = np.prod(X[:, indices], axis=1, keepdims=True)
                        features.append(feature)
            else:
                # All polynomial terms
                from itertools import combinations_with_replacement
                for indices in combinations_with_replacement(range(n_features), deg):
                    feature = np.prod(X[:, indices], axis=1, keepdims=True)
                    features.append(feature)
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
