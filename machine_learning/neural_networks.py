"""
Neural Network Implementation

Implements feedforward neural networks with backpropagation for
supervised learning tasks.
"""

import math
import random
from typing import List, Callable, Tuple


class Neuron:
    """Individual neuron with weighted inputs and activation function."""
    
    def __init__(self, num_inputs: int, activation: str = 'sigmoid'):
        """
        Initialize neuron with random weights.
        
        Args:
            num_inputs: Number of input connections
            activation: Activation function ('sigmoid', 'tanh', 'relu')
        """
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_name = activation
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        elif activation == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # For backpropagation
        self.last_input = None
        self.last_output = None
        self.delta = 0.0
    
    def forward(self, inputs: List[float]) -> float:
        """
        Forward pass through neuron.
        
        Args:
            inputs: Input values
            
        Returns:
            Neuron output after activation
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        
        # Calculate weighted sum
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Apply activation function
        output = self.activation(weighted_sum)
        
        # Store for backpropagation
        self.last_input = inputs.copy()
        self.last_output = output
        
        return output
    
    def backward(self, error: float, learning_rate: float) -> List[float]:
        """
        Backward pass for weight updates.
        
        Args:
            error: Error signal for this neuron
            learning_rate: Learning rate for weight updates
            
        Returns:
            Error signals for previous layer
        """
        # Calculate delta
        self.delta = error * self.activation_derivative(self.last_output)
        
        # Calculate input errors
        input_errors = [self.delta * w for w in self.weights]
        
        # Update weights and bias
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.delta * self.last_input[i]
        self.bias -= learning_rate * self.delta
        
        return input_errors
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def _sigmoid_derivative(self, y: float) -> float:
        """Derivative of sigmoid function."""
        return y * (1.0 - y)
    
    def _tanh(self, x: float) -> float:
        """Hyperbolic tangent activation function."""
        return math.tanh(x)
    
    def _tanh_derivative(self, y: float) -> float:
        """Derivative of tanh function."""
        return 1.0 - y * y
    
    def _relu(self, x: float) -> float:
        """ReLU activation function."""
        return max(0.0, x)
    
    def _relu_derivative(self, y: float) -> float:
        """Derivative of ReLU function."""
        return 1.0 if y > 0 else 0.0


class Layer:
    """Neural network layer containing multiple neurons."""
    
    def __init__(self, num_neurons: int, num_inputs: int, activation: str = 'sigmoid'):
        """
        Initialize layer with neurons.
        
        Args:
            num_neurons: Number of neurons in layer
            num_inputs: Number of inputs to each neuron
            activation: Activation function for all neurons
        """
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_neurons)]
        self.last_output = None
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass through layer.
        
        Args:
            inputs: Input values
            
        Returns:
            Layer outputs
        """
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        self.last_output = outputs
        return outputs
    
    def backward(self, errors: List[float], learning_rate: float) -> List[float]:
        """
        Backward pass through layer.
        
        Args:
            errors: Error signals for each neuron
            learning_rate: Learning rate for updates
            
        Returns:
            Error signals for previous layer
        """
        if len(errors) != len(self.neurons):
            raise ValueError(f"Expected {len(self.neurons)} errors, got {len(errors)}")
        
        # Collect input errors from all neurons
        input_errors = []
        for i, neuron in enumerate(self.neurons):
            neuron_input_errors = neuron.backward(errors[i], learning_rate)
            
            # Initialize input_errors for first neuron
            if i == 0:
                input_errors = neuron_input_errors.copy()
            else:
                # Sum errors from all neurons
                for j in range(len(neuron_input_errors)):
                    input_errors[j] += neuron_input_errors[j]
        
        return input_errors


class NeuralNetwork:
    """
    Feedforward Neural Network with backpropagation.
    
    Implements a multi-layer perceptron that can be used for
    classification and regression tasks.
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None,
                 learning_rate: float = 0.1):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: Activation functions for each layer (except input)
            learning_rate: Learning rate for training
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least 2 layers")
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        # Set default activations
        if activations is None:
            activations = ['sigmoid'] * (len(layer_sizes) - 1)
        
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must match number of layers minus 1")
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i + 1], layer_sizes[i], activations[i])
            self.layers.append(layer)
        
        # Training statistics
        self.training_error = []
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward propagation through network.
        
        Args:
            inputs: Input values
            
        Returns:
            Network outputs
        """
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {len(inputs)}")
        
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        
        return outputs
    
    def backward(self, target: List[float]) -> float:
        """
        Backward propagation for training.
        
        Args:
            target: Target output values
            
        Returns:
            Mean squared error for this sample
        """
        # Get current output
        current_output = self.layers[-1].last_output
        
        if len(target) != len(current_output):
            raise ValueError(f"Expected {len(current_output)} targets, got {len(target)}")
        
        # Calculate output layer errors
        output_errors = [target[i] - current_output[i] for i in range(len(target))]
        
        # Calculate mean squared error
        mse = sum(error ** 2 for error in output_errors) / len(output_errors)
        
        # Backpropagate errors
        errors = output_errors
        for layer in reversed(self.layers):
            errors = layer.backward(errors, self.learning_rate)
        
        return mse
    
    def train(self, X: List[List[float]], y: List[List[float]], 
              epochs: int = 1000, verbose: bool = False) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of training epochs
            verbose: Print training progress
            
        Returns:
            List of training errors per epoch
        """
        if len(X) != len(y):
            raise ValueError("Number of inputs must match number of targets")
        
        self.training_error = []
        
        for epoch in range(epochs):
            epoch_error = 0.0
            
            for inputs, targets in zip(X, y):
                # Forward pass
                self.forward(inputs)
                
                # Backward pass
                error = self.backward(targets)
                epoch_error += error
            
            # Average error for epoch
            avg_error = epoch_error / len(X)
            self.training_error.append(avg_error)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {avg_error:.6f}")
        
        return self.training_error
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """
        Make predictions on new data.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Network predictions
        """
        predictions = []
        for inputs in X:
            output = self.forward(inputs)
            predictions.append(output)
        
        return predictions
    
    def predict_single(self, inputs: List[float]) -> List[float]:
        """
        Make prediction on single input.
        
        Args:
            inputs: Single input vector
            
        Returns:
            Network output
        """
        return self.forward(inputs)


class MLPClassifier(NeuralNetwork):
    """
    Multi-layer Perceptron for classification tasks.
    
    Extends NeuralNetwork with classification-specific methods.
    """
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100,),
                 activation: str = 'relu', learning_rate: float = 0.001,
                 max_iter: int = 200):
        """
        Initialize MLP classifier.
        
        Args:
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function for hidden layers
            learning_rate: Learning rate for training
            max_iter: Maximum number of iterations
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.classes_ = None
        self.n_features_ = None
        self.n_outputs_ = None
        
        # Will be initialized in fit()
        super().__init__([1], [activation], learning_rate)
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'MLPClassifier':
        """
        Fit the MLP classifier.
        
        Args:
            X: Training data
            y: Target class labels
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        # Determine classes and dimensions
        self.classes_ = sorted(list(set(y)))
        self.n_features_ = len(X[0])
        self.n_outputs_ = len(self.classes_)
        
        # Create layer sizes: input -> hidden layers -> output
        layer_sizes = [self.n_features_] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        
        # Create activations: hidden layers use specified activation, output uses sigmoid
        activations = [self.activation] * len(self.hidden_layer_sizes) + ['sigmoid']
        
        # Reinitialize with correct architecture
        super().__init__(layer_sizes, activations, self.learning_rate)
        
        # Convert labels to one-hot encoding
        y_onehot = []
        for label in y:
            onehot = [0.0] * self.n_outputs_
            class_index = self.classes_.index(label)
            onehot[class_index] = 1.0
            y_onehot.append(onehot)
        
        # Train the network
        self.train(X, y_onehot, self.max_iter, verbose=False)
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for inputs in X:
            outputs = self.forward(inputs)
            # Find class with highest probability
            predicted_class_index = outputs.index(max(outputs))
            predicted_class = self.classes_[predicted_class_index]
            predictions.append(predicted_class)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities for each sample
        """
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet")
        
        probabilities = []
        for inputs in X:
            outputs = self.forward(inputs)
            probabilities.append(outputs)
        
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


class MLPRegressor(NeuralNetwork):
    """
    Multi-layer Perceptron for regression tasks.
    
    Extends NeuralNetwork with regression-specific methods.
    """
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100,),
                 activation: str = 'relu', learning_rate: float = 0.001,
                 max_iter: int = 200):
        """
        Initialize MLP regressor.
        
        Args:
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function for hidden layers
            learning_rate: Learning rate for training
            max_iter: Maximum number of iterations
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.n_features_ = None
        self.n_outputs_ = None
        
        # Will be initialized in fit()
        super().__init__([1], [activation], learning_rate)
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'MLPRegressor':
        """
        Fit the MLP regressor.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        # Determine dimensions
        self.n_features_ = len(X[0])
        self.n_outputs_ = 1  # Single output for regression
        
        # Create layer sizes: input -> hidden layers -> output
        layer_sizes = [self.n_features_] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        
        # Create activations: hidden layers use specified activation, output is linear (sigmoid)
        activations = [self.activation] * len(self.hidden_layer_sizes) + ['sigmoid']
        
        # Reinitialize with correct architecture
        super().__init__(layer_sizes, activations, self.learning_rate)
        
        # Convert targets to list of lists for consistency
        y_formatted = [[target] for target in y]
        
        # Train the network
        self.train(X, y_formatted, self.max_iter, verbose=False)
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict target values.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        if self.n_features_ is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for inputs in X:
            outputs = self.forward(inputs)
            predictions.append(outputs[0])  # Single output value
        
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
    # Test basic neural network
    print("Testing basic neural network...")
    
    # XOR problem
    X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_xor = [[0], [1], [1], [0]]
    
    nn = NeuralNetwork([2, 4, 1], ['sigmoid', 'sigmoid'], learning_rate=0.5)
    errors = nn.train(X_xor, y_xor, epochs=1000, verbose=True)
    
    print("\nXOR predictions:")
    for inputs in X_xor:
        output = nn.predict_single(inputs)
        print(f"{inputs} -> {output[0]:.3f}")
    
    # Test MLP classifier
    print("\nTesting MLP classifier...")
    
    # Simple 2D classification problem
    X_class = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]
    y_class = [0, 1, 1, 0, 1]
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), learning_rate=0.1, max_iter=500)
    clf.fit(X_class, y_class)
    
    predictions = clf.predict(X_class)
    accuracy = clf.score(X_class, y_class)
    
    print(f"Classification accuracy: {accuracy:.3f}")
    print(f"Predictions: {predictions}")
    
    # Test MLP regressor
    print("\nTesting MLP regressor...")
    
    # Simple regression problem: y = x1 + x2
    X_reg = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_reg = [3, 5, 7, 9]
    
    reg = MLPRegressor(hidden_layer_sizes=(5,), learning_rate=0.01, max_iter=1000)
    reg.fit(X_reg, y_reg)
    
    reg_predictions = reg.predict(X_reg)
    r2_score = reg.score(X_reg, y_reg)
    
    print(f"Regression R² score: {r2_score:.3f}")
    print(f"Predictions: {[f'{p:.3f}' for p in reg_predictions]}")
    print(f"True values: {y_reg}")
