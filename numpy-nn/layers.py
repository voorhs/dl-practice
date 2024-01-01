from scipy.special import expit
import numpy as np
from scipy.special import logsumexp


class ReLU:
    """ReLU layer simply applies elementwise rectified linear unit to all inputs"""

    def __init__(self):
        self.params = []  # ReLU has no parameters

    def forward(self, input):
        """Input shape: (batch, num_units)"""
        self.mask = input > 0
        return np.where(self.mask, input, 0)

    def backward(self, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        grad_output shape: (batch, num_units)
        output 1 shape: (batch, num_units)
        output 2: []
        """
        return self.mask * grad_output, []

    def __repr__(self):
        return 'Relu()'


class Tanh:
    """
    tanh(y) = (e^y - e^(-y)) / (e^y + e^(-y))
    """

    def __init__(self):
        self.params = []  # Tanh has no parameters

    def forward(self, input):
        """
        Apply elementwise Tanh to [batch, num_units] matrix
        """
        self.input = input.copy()
        return np.tanh(input)

    def backward(self, grad_output):
        """
        Compute gradient of loss w.r.t. Tanh input
        grad_output shape: [batch, num_units]
        output 1 shape: [batch, num_units]
        output 2: []
        """
        return 4 * expit(2 * self.input) ** 2 * (1 - expit(2 * self.input)) * grad_output, []

    def __repr__(self):
        return 'Tanh()'


class Sigmoid:
    """
    sigmoid(y) = 1 / (1 + e^(-y))
    """

    def __init__(self):
        self.params = []  # Sigmoid has no parameters

    def forward(self, input):
        """
        Apply elementwise Sigmoid to [batch, num_units] matrix
        """
        self.input = input.copy()
        return expit(input)

    def backward(self, grad_output):
        """
        Compute gradient of loss w.r.t. Sigmoid input
        grad_output shape: [batch, num_units]
        output 1 shape: [batch, num_units]
        output 2: []
        """
        return expit(self.input) * (1 - expit(self.input)) * grad_output, []

    def __repr__(self):
        return 'Sigmoid()'


class Dense:
    def __init__(self, input_units, output_units):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = W x + b
        """
        # initialize weights with small random numbers from normal distribution
        self.weights = np.random.randn(output_units, input_units) * 0.01
        self.biases = np.zeros(output_units)
        self.params = [self.weights, self.biases]

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = W x + b

        input shape: (batch, input_units)
        output shape: (batch, output units)
        """
        self.input = input.copy()
        return input @ self.weights.T + self.biases[None, :]

    def backward(self, grad_output):
        """
        Compute gradients
        grad_output shape: (batch, output_units)
        output shapes: (batch, input_units), (num_params,)
        """
        grad_input = grad_output @ self.weights
        grad_weights = grad_output.T @ self.input
        grad_biases = grad_output.sum(axis=0)
        return grad_input, [np.r_[grad_weights.ravel(), grad_biases]]

    def __repr__(self):
        return f'Dense({self.weights.shape[1]}, {self.weights.shape[0]})'


class LogSoftmax:
    def __init__(self, n_in):
        self.params = []
        self.n_in = n_in

    def forward(self, input):
        """
        Applies softmax to each row and then applies component-wise log
        Input shape: (batch, num_units)
        Output shape: (batch, num_units)
        """
        self.output = input - logsumexp(input, axis=1)[:, None]
        return self.output

    def backward(self, grad_output):
        """
        Input shape: (batch, num_units)
        Output shape: (batch, num_units), []
        """
        grad_input = grad_output - \
            grad_output.sum(axis=1, keepdims=True) * np.exp(self.output)
        return grad_input, []

    def __repr__(self):
        return f'LogSoftmax({self.n_in})'
