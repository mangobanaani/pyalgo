# Machine Learning Algorithms

This directory contains implementations of fundamental machine learning algorithms for supervised and unsupervised learning tasks. All algorithms are implemented in pure Python without external dependencies, making them educational and self-contained.

## Available Algorithms

### Supervised Learning

**Linear Regression** (`linear_regression.py`)
- Standard Linear Regression with normal equation and SVD methods
- Ridge Regression with L2 regularization to prevent overfitting
- Lasso Regression with L1 regularization for feature selection
- Feature normalization and scaling support
- Multiple evaluation metrics (R², MSE, MAE)

**Neural Networks** (`neural_networks.py`)
- Multi-layer Perceptron (MLP) with customizable architecture
- Multiple activation functions: Sigmoid, Tanh, ReLU
- Backpropagation algorithm for training
- MLPClassifier for classification tasks
- MLPRegressor for regression tasks
- Configurable learning rates and network topology

**Decision Trees** (`decision_trees.py`)
- DecisionTreeClassifier with Gini impurity and entropy criteria
- DecisionTreeRegressor using mean squared error reduction
- Configurable tree depth and minimum samples constraints
- Automatic feature selection and threshold optimization
- Pruning capabilities to prevent overfitting

**Support Vector Machines** (`svm.py`)
- SVM classifier using Sequential Minimal Optimization (SMO)
- Multiple kernel functions: Linear, Polynomial, RBF, Sigmoid
- Support Vector Regression (SVR) for continuous targets
- Automatic support vector identification
- Configurable regularization parameters

**Ensemble Methods** (`ensemble_methods.py`)
- Random Forest with bootstrap aggregating and feature randomness
- AdaBoost (Adaptive Boosting) with weak learner integration
- Configurable number of estimators and tree parameters
- Voting mechanisms for improved prediction accuracy

### Unsupervised Learning

**K-Means Clustering** (`kmeans.py`)
- Standard K-Means with k-means++ initialization
- Configurable number of clusters and convergence criteria
- Cluster center computation and assignment
- Inertia calculation for cluster quality assessment
- Support for different distance metrics

## Key Features

### Algorithm Characteristics

- **Pure Python Implementation**: No external dependencies required
- **Educational Focus**: Well-commented code with clear algorithm explanations
- **Scikit-learn Compatible API**: Familiar fit/predict interface
- **Robust Error Handling**: Input validation and edge case management
- **Performance Tracking**: Training history and convergence monitoring

### Common Interface

All algorithms follow a consistent interface pattern:

```python
# Create model instance
model = Algorithm(parameters...)

# Fit to training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
score = model.score(X_test, y_test)
```

## Usage Examples

### Linear Regression
```python
from machine_learning import LinearRegression, Ridge, Lasso

# Standard linear regression
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
r2_score = lr.score(X_test, y_test)

# Ridge regression with regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso regression for feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

### Neural Networks
```python
from machine_learning import MLPClassifier, MLPRegressor

# Classification
clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    learning_rate=0.001,
    max_iter=500
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Regression
reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    learning_rate=0.01
)
reg.fit(X_train, y_train)
```

### Decision Trees
```python
from machine_learning import DecisionTreeClassifier, DecisionTreeRegressor

# Classification tree
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
clf.fit(X_train, y_train)

# Regression tree
reg = DecisionTreeRegressor(
    max_depth=8,
    min_samples_split=10
)
reg.fit(X_train, y_train)
```

### Ensemble Methods
```python
from machine_learning import RandomForestClassifier, AdaBoostClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    bootstrap=True
)
rf.fit(X_train, y_train)

# AdaBoost
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X_train, y_train)
```

### Support Vector Machines
```python
from machine_learning import SVM, SVR

# SVM Classification
svm = SVM(
    C=1.0,
    kernel='rbf',
    gamma='scale'
)
svm.fit(X_train, y_train)

# SVR Regression
svr = SVR(
    C=1.0,
    epsilon=0.1,
    kernel='rbf'
)
svr.fit(X_train, y_train)
```

### K-Means Clustering
```python
from machine_learning import KMeans

# K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=42
)
cluster_labels = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_
inertia = kmeans.inertia_
```

## Algorithm Selection Guide

### Problem Type

**Regression Tasks**:
- Linear problems: LinearRegression, Ridge, Lasso
- Non-linear patterns: MLPRegressor, DecisionTreeRegressor
- High-dimensional data: SVR with RBF kernel
- Robust predictions: Random Forest (not yet implemented)

**Classification Tasks**:
- Linear separable: LinearSVM, MLPClassifier
- Non-linear boundaries: SVM with RBF kernel, DecisionTreeClassifier
- Ensemble predictions: RandomForestClassifier, AdaBoostClassifier
- Probability estimates: MLPClassifier, DecisionTreeClassifier

**Clustering Tasks**:
- Spherical clusters: KMeans
- Different cluster shapes: (Future: DBSCAN, Hierarchical)

### Data Characteristics

**Small Datasets (< 1000 samples)**:
- DecisionTreeClassifier/Regressor
- SVM with any kernel
- MLPClassifier/Regressor with small networks

**Large Datasets (> 10000 samples)**:
- LinearRegression with normalization
- KMeans clustering
- MLPClassifier/Regressor with stochastic training

**High-Dimensional Data**:
- Lasso for feature selection
- SVM with linear kernel
- PCA preprocessing (future implementation)

**Noisy Data**:
- Ridge regression for regularization
- Random Forest for robustness
- SVM with appropriate C parameter

## Performance Considerations

### Computational Complexity

- **Linear Regression**: O(n³) for normal equation, O(nmp) for gradient descent
- **K-Means**: O(nkmi) where n=samples, k=clusters, m=features, i=iterations
- **Decision Trees**: O(nm log n) for building, O(log n) for prediction
- **Neural Networks**: O(nmhle) where h=hidden units, l=layers, e=epochs
- **SVM**: O(n²) to O(n³) depending on kernel and optimization

### Memory Usage

- **Linear models**: O(m) for coefficients
- **Tree models**: O(nodes) proportional to tree size
- **Neural Networks**: O(weights) proportional to network size
- **SVM**: O(support_vectors) typically much less than training size

## Model Evaluation

### Classification Metrics
- Accuracy: Proportion of correct predictions
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall

### Regression Metrics
- R² Score: Coefficient of determination (1 = perfect fit)
- Mean Squared Error (MSE): Average squared differences
- Mean Absolute Error (MAE): Average absolute differences
- Root Mean Squared Error (RMSE): Square root of MSE

### Clustering Metrics
- Inertia: Within-cluster sum of squared distances
- Silhouette Score: Measure of cluster separation
- Adjusted Rand Index: Similarity to ground truth labels

## Implementation Details

### Design Principles

1. **Educational Value**: Code is written to be readable and educational
2. **Algorithmic Accuracy**: Implementations follow standard algorithms
3. **Practical Usability**: Real-world applicable with proper error handling
4. **Extensibility**: Easy to modify and extend for custom needs

### Numerical Stability

- SVD decomposition for linear regression when matrices are ill-conditioned
- Numerical checks for division by zero and overflow conditions
- Gradient clipping in neural networks to prevent exploding gradients
- Regularization techniques to improve numerical stability

### Testing and Validation

Each algorithm includes:
- Unit tests for core functionality
- Integration tests with realistic datasets
- Performance benchmarks on standard problems
- Edge case handling validation

## Future Enhancements

### Planned Additions

- **Logistic Regression**: For binary and multiclass classification
- **Naive Bayes**: Probabilistic classifier for text and categorical data
- **K-Nearest Neighbors**: Instance-based learning algorithm
- **Hierarchical Clustering**: Agglomerative and divisive clustering
- **DBSCAN**: Density-based clustering for irregular shapes
- **Principal Component Analysis**: Dimensionality reduction
- **Cross-Validation**: Model selection and hyperparameter tuning

### Advanced Features

- **Model Persistence**: Save and load trained models
- **Feature Engineering**: Automatic feature creation and selection
- **Hyperparameter Optimization**: Grid search and random search
- **Ensemble Combinations**: Voting and stacking methods
- **Online Learning**: Incremental training for streaming data
