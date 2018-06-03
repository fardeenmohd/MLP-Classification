from data.data_class import DataClass, DataType
from numpy import random, exp, dot


class Layer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.input = None


class MultiLayerPerceptron:
    def __init__(self, data_type: DataType = DataType.POKER_HANDS, number_of_hidden_layers: int = 2,
                 hidden_layer_size: int = 20, data_size: int = 1500, training_percentage: int = 0.8):
        self.data_type = data_type
        self.number_of_hidden_layers = number_of_hidden_layers
        self.data_class = DataClass(data_type, data_size, training_percentage)
        self.hidden_layer_size = hidden_layer_size
        self.layers = []

        self.layers.append(Layer(hidden_layer_size, self.data_class.features_x))  # input layer added
        for i in range(self.number_of_hidden_layers):
            if i == self.number_of_hidden_layers - 1:
                self.layers.append(Layer(self.data_class.features_y, self.hidden_layer_size))
            else:
                self.layers.append(Layer(self.hidden_layer_size, hidden_layer_size))

        for i in range(self.layers.__len__()):
            print("Layer: " + i.__str__() + " has weights with shape: " + self.layers[i].weights.shape.__str__())

    def propagate_forward(self, inputs):
        self.layers[0].input = inputs
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.sigmoid(dot(self.layers[i - 1].input, self.layers[i - 1].weights))

        return self.layers[-1].input

    def propagate_backward(self, target, learning_rate: int):
        # Compute error on output layer
        deltas = []
        error = target - self.layers[-1].input
        delta = error * self.sigmoid_derivative(self.layers[-1].input)
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            delta = dot(deltas[0], self.layers[i].weights.T) * self.sigmoid_derivative(self.layers[i].input)
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.layers) - 1):
            layer = self.layers[i].input
            delta = deltas[i]
            dw = dot(layer.T, delta)
            self.layers[i].weights += learning_rate * dw

    def train(self, epochs: int = 1, learning_rate: int = 0.1):
        for i in range(epochs):
            self.propagate_forward(self.data_class.train_x)
            self.propagate_backward(self.data_class.train_y, learning_rate)

    def test(self):
        output = self.propagate_forward(self.data_class.test_x)
        expected = self.data_class.test_y
        print("output: " + output.shape.__str__())
        print("expected: " + expected.shape.__str__())

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
