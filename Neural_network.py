import numpy as np
import json


class NeuralNetwork:
    """ Class of Neural Network

    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """ Init method to create instance of Neural Network

        :param input_size: Number of inputs
        :param hidden_layer_sizes: NUmber of hidden layers
        :param output_size: Number of outputs
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activations = []

        # Assign random values for initial weights
        sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = [np.random.randn(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    @staticmethod
    def load_configuration():
        """ Method toi load the initial configurations

        :return: Returns object containing configuration values
        """
        with open('configuration.json', 'r') as file:
            config_values = json.load(file)
        return config_values

    def forward(self, inputs):
        """ Method to implement the forward propagation

        :param inputs: Input values
        :return: List of the activations
        """

        self.activations = [inputs]
        for i, w in enumerate(self.weights):
            input_activation = np.dot(self.activations[i], w)
            output_activation = self.sigmoid(input_activation)
            self.activations.append(output_activation)
        return self.activations[-1]

    def backward(self, targets, learning_rate):
        """ Method to apply backward propagation

        :param targets: Target values
        :param learning_rate: Value to determine how much the model will change in response to the calculated error
        :return:
        """
        # Backward propagation
        output_errors = targets - self.activations[-1]
        deltas = [output_errors * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] += np.dot(self.activations[i].T, deltas[i]) * learning_rate

    @staticmethod
    def sigmoid(x):
        """ Method to define sigmoid function

        :param x: Value to be calculated
        :return: Result of the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """ Method to calculate the derivative of the sigmoid function to be used on teh backward propagation

        :param x: Value to be calculated
        :return:
        """
        return x * (1 - x)

    def mean_squared_error(self, targets):
        """ Method to calculate the mean squared error

        :param targets:
        :return: Returns the error value
        """
        return np.mean(np.square(targets - self.activations[-1]))

    def get_weights(self):
        """ Method to get the weights between nodes

        :return: List of all the weights between nodes
        """
        return [w.tolist() for w in self.weights]

    def get_activations(self):
        """ Method to get the values of the activations

        :return: List of all the activation values
        """
        return [a.tolist() for a in self.activations]
