"""
Machine Learning Algorithms

This package implements fundamental machine learning algorithms including
supervised learning, unsupervised learning, and model evaluation.
"""

from .linear_regression import LinearRegression, Ridge, Lasso
from .kmeans import KMeans
from .neural_networks import NeuralNetwork, MLPClassifier, MLPRegressor
from .decision_trees import DecisionTreeClassifier, DecisionTreeRegressor
from .ensemble_methods import RandomForestClassifier, AdaBoostClassifier
from .svm import SVM, SVR

__all__ = [
    'LinearRegression',
    'Ridge', 
    'Lasso',
    'KMeans',
    'NeuralNetwork',
    'MLPClassifier',
    'MLPRegressor',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'SVM',
    'SVR'
]

# Supervised Learning
from .linear_regression import LinearRegression, Ridge, Lasso, ElasticNet
from .logistic_regression import LogisticRegression, MultinomialLogisticRegression
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .svm import SupportVectorMachine, SVMClassifier, SVMRegressor
from .naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes
from .knn import KNearestNeighbors, KNNClassifier, KNNRegressor

# Unsupervised Learning
from .kmeans import KMeans, KMeansPlus
from .hierarchical_clustering import AgglomerativeClustering, DivisiveClustering
from .dbscan import DBSCAN
from .pca import PrincipalComponentAnalysis
from .gaussian_mixture import GaussianMixtureModel

# Neural Networks
from .neural_network import NeuralNetwork, MultiLayerPerceptron
from .activation_functions import Sigmoid, ReLU, Tanh, Softmax, LeakyReLU
from .loss_functions import MeanSquaredError, CrossEntropy, BinaryCrossEntropy
from .optimizers import SGD, Adam, RMSprop, AdaGrad

# Ensemble Methods
from .bagging import BaggingClassifier, BaggingRegressor
from .boosting import AdaBoost, GradientBoosting
from .voting import VotingClassifier, VotingRegressor

# Model Evaluation
from .metrics import accuracy_score, precision_score, recall_score, f1_score
from .metrics import mean_squared_error, r2_score, confusion_matrix
from .cross_validation import KFoldCV, StratifiedKFoldCV, LeaveOneOutCV
from .model_selection import train_test_split, GridSearchCV

# Feature Engineering
from .preprocessing import StandardScaler, MinMaxScaler, Normalizer
from .feature_selection import SelectKBest, RecursiveFeatureElimination
from .dimensionality_reduction import PCA, LDA, TSNE

__all__ = [
    # Supervised Learning
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
    'LogisticRegression', 'MultinomialLogisticRegression',
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    'RandomForestClassifier', 'RandomForestRegressor',
    'SupportVectorMachine', 'SVMClassifier', 'SVMRegressor',
    'GaussianNaiveBayes', 'MultinomialNaiveBayes',
    'KNearestNeighbors', 'KNNClassifier', 'KNNRegressor',
    
    # Unsupervised Learning
    'KMeans', 'KMeansPlus',
    'AgglomerativeClustering', 'DivisiveClustering',
    'DBSCAN',
    'PrincipalComponentAnalysis',
    'GaussianMixtureModel',
    
    # Neural Networks
    'NeuralNetwork', 'MultiLayerPerceptron',
    'Sigmoid', 'ReLU', 'Tanh', 'Softmax', 'LeakyReLU',
    'MeanSquaredError', 'CrossEntropy', 'BinaryCrossEntropy',
    'SGD', 'Adam', 'RMSprop', 'AdaGrad',
    
    # Ensemble Methods
    'BaggingClassifier', 'BaggingRegressor',
    'AdaBoost', 'GradientBoosting',
    'VotingClassifier', 'VotingRegressor',
    
    # Model Evaluation
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'mean_squared_error', 'r2_score', 'confusion_matrix',
    'KFoldCV', 'StratifiedKFoldCV', 'LeaveOneOutCV',
    'train_test_split', 'GridSearchCV',
    
    # Feature Engineering
    'StandardScaler', 'MinMaxScaler', 'Normalizer',
    'SelectKBest', 'RecursiveFeatureElimination',
    'PCA', 'LDA', 'TSNE'
]
