from data.data_class import DataClass, DataType
from numpy import random, exp, dot


class Layer:
    def __init__(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons
        self.input = None


class MultiLayerPerceptron:
    def __init__(self, data_type: DataType = DataType.POKER_HANDS, number_of_hidden_layers: int = 2,
                 hidden_layer_size: int = 100, data_size: int = 1500, training_percentage: int = 0.5):
        self.data_type = data_type
        self.number_of_hidden_layers = number_of_hidden_layers
        self.data_class = DataClass(data_type, data_size, training_percentage)
        self.hidden_layer_size = hidden_layer_size
        self.layers = []

        self.layers.append(Layer(self.data_class.features_x))  # input layer added
        for i in range(self.number_of_hidden_layers):  # hidden layers added
            self.layers.append(Layer(self.hidden_layer_size))
        self.layers.append(Layer(self.data_class.features_y))  # output layer added

        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append(self.create_weights(self.layers[i].number_of_neurons,
                                                    self.layers[i + 1].number_of_neurons))

    def print_config(self):
        for i in range(len(self.layers)):
            print("Layer: " + i.__str__() + " has number of neurons: " + self.layers[i].number_of_neurons.__str__())

        for i in range(len(self.layers) - 1):
            print("Weight matrix between Layer " + i.__str__() + " and " + (i + 1).__str__() +
                  " has a shape: " + self.weights[i].shape.__str__())

    def propagate_forward(self, inputs):
        self.layers[0].input = inputs
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.sigmoid(dot(self.layers[i - 1].input, self.weights[i - 1]))

        return self.layers[-1].input

    def propagate_backward(self, target, learning_rate: int):
        # Compute error on output layer
        deltas = []
        error = target - self.layers[-1].input
        delta = error * self.sigmoid_derivative(self.layers[-1].input)
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            delta = dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(self.layers[i].input)
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.layers) - 1):
            layer = self.layers[i].input
            delta = deltas[i]
            dw = dot(layer.T, delta)
            self.weights[i] += learning_rate * dw

        return (error ** 2).sum()

    def train(self, epochs: int = 10, learning_rate: int = 1):
        self.print_config()
        for i in range(epochs):
            self.propagate_forward(self.data_class.train_x)
            self.propagate_backward(self.data_class.train_y, learning_rate)

    def test(self):
        output = self.propagate_forward(self.data_class.test_x)
        expected = self.data_class.test_y

        error = ((expected - output) ** 2).sum()
        print("Error after testing with trained weights: " + error.__str__())
        return error

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod
    def create_weights(number_of_neurons, number_of_inputs_per_neuron):
        return 2 * random.random((number_of_neurons, number_of_inputs_per_neuron)) - 1
