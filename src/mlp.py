from data.data_class import DataClass, DataType
from numpy import random, exp, dot, array, zeros_like


class Layer:
    def __init__(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons
        self.input = None


class MultiLayerPerceptron:
    def __init__(self, data_type: DataType = DataType.POKER_HANDS, number_of_hidden_layers: int = 2,
                 hidden_layer_size: int = 20, data_size: int = 1500, training_percentage: int = 0.8):
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

        self.print_config()

    def print_config(self):
        print("----------------Neural-Network-Specification-----------------")
        print("Data class selected: " + self.data_type.value.__str__())
        for i in range(len(self.layers)):
            print("Layer: " + i.__str__() + " has number of neurons: " + self.layers[i].number_of_neurons.__str__())

        for i in range(len(self.layers) - 1):
            print("Weight matrix between Layer " + i.__str__() + " and " + (i + 1).__str__() +
                  " has a shape: " + self.weights[i].shape.__str__())

        print("Number of features used: " + str(self.data_class.features_x) + ", Number of features classified: " + str(
            self.data_class.features_y))
        print("-------------------------------------------------------------")

    def propagate_forward(self, inputs):
        self.layers[0].input = inputs
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.sigmoid(dot(self.layers[i - 1].input, self.weights[i - 1]))

        return self.layers[-1].input

    def propagate_backward(self, target, learning_rate: int):
        # Compute error on output layer
        initial_error = target - self.layers[-1].input
        error = initial_error
        # Compute error on hidden layers
        for i in range(len(self.layers) - 1, 0, -1):
            delta = error * self.sigmoid_derivative(self.layers[i].input)
            error = delta.dot(self.weights[i - 1].T)
            grad = self.layers[i - 1].input.T.dot(delta)
            self.weights[i - 1] += learning_rate * grad

    def train(self, epochs: int = 10, learning_rate: int = 1):
        for i in range(epochs):
            self.propagate_forward(self.data_class.train_x)
            self.propagate_backward(self.data_class.train_y, learning_rate)
        classified_tr_output = self.classify(self.propagate_forward(self.data_class.train_x))
        num_train_matches = 0
        for i in range(self.data_class.train_x.shape[0]):
            if (classified_tr_output[i] == self.data_class.train_y[i]).all():
                num_train_matches += 1
        success_rate = 100 * (num_train_matches / self.data_class.train_row_count)
        print("Success rate of your model in training phase is: " + str(success_rate) + "%")
        return success_rate

    def test(self):
        classified_te_output = self.classify(self.propagate_forward(self.data_class.test_x))
        num_test_matches = 0
        for i in range(self.data_class.test_x.shape[0]):
            if (classified_te_output[i] == self.data_class.test_y[i]).all():
                num_test_matches += 1
        success_rate = 100 * (num_test_matches / self.data_class.test_row_count)
        print("Success rate of your model in testing phase is: " + str(success_rate) + "%")
        return success_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod
    def create_weights(number_of_neurons, number_of_inputs_per_neuron):
        return 2 * random.random((number_of_neurons, number_of_inputs_per_neuron)) - 1

    @staticmethod
    def classify(result: array):
        tmp = zeros_like(result)
        tmp[range(len(result)), result.argmax(axis=1)] = 1
        return tmp
