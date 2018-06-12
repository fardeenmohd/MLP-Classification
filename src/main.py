from argparse import ArgumentParser
from enum import Enum

from data.data_class import DataType
from src.mlp import MultiLayerPerceptron
import matplotlib.pyplot as plt


class TestType(Enum):
    LAYER_SIZE = 'LayerSize'
    HIDDEN_LAYERS = 'HiddenLayers'
    ITERATION = 'Iterations'
    LEARNING_RATE = 'LearningRate'


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-d', '--DataType', default=DataType.POKER_HANDS.value, type=str,
                        help='WhiteWine, RedWine, DrugConsumption or PokerHands')
    parser.add_argument('-ds', '--DataSize', default=2000, type=int, help='Size of data')
    parser.add_argument('-l', '--HiddenLayers', default=2, type=int, help='Number of hidden layers')
    parser.add_argument('-ls', '--HiddenLayerSize', default=20, type=int, help='Size of the hidden layer')
    parser.add_argument('-tp', '--TrainingPercentage', default=0.8, type=int, help='Training Percentage')
    parser.add_argument('-i', '--Iterations', default=100, type=int, help='Number of epochs')
    parser.add_argument('-lr', '--LearningRate', default=1, type=float, help='Learning rate')
    parser.add_argument('-t', '--Test', type=str, default=None,
                        help='LayerSize, HiddenLayers, LearningRate or Iterations')
    parser.add_argument('-str', '--Start', type=int, default=1, help='Starting point for testing')
    parser.add_argument('-stp', '--Stop', type=int, default=1000, help='Stopping point for testing')
    parser.add_argument('-st', '--Step', type=int, default=100, help='Steps for testing')

    arguments = vars(parser.parse_args())
    parser.print_help()
    print(arguments)
    data_arg = arguments['DataType']
    data_type = DataType.POKER_HANDS

    if data_arg is DataType.WHITE_WINE.value:
        data_type = DataType.WHITE_WINE
    elif data_arg is DataType.RED_WINE.value:
        data_type = DataType.RED_WINE
    elif data_arg is DataType.DRUG_CONSUMPTION.value:
        data_type = DataType.DRUG_CONSUMPTION
    else:
        data_type = DataType.POKER_HANDS

    if arguments['Test']:
        test_type = arguments['Test']
        testing_parameters = []
        test_success_rates = []
        train_success_rates = []

        start = arguments['Start']
        stop = arguments['Stop']
        step = arguments['Step']

        for i in range(start, stop, step):
            testing_parameters.append(i)

        if test_type == TestType.HIDDEN_LAYERS.value:
            for testing_parameter in testing_parameters:
                mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=testing_parameter,
                                           hidden_layer_size=arguments['HiddenLayerSize'],
                                           data_size=arguments['DataSize'],
                                           training_percentage=arguments['TrainingPercentage'])
                train_success_rates.append(
                    mlp.train(epochs=arguments['Iterations'], learning_rate=arguments['LearningRate']))
                test_success_rates.append(mlp.test())

        elif test_type == TestType.LAYER_SIZE.value:
            for testing_parameter in testing_parameters:
                mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=arguments['HiddenLayers'],
                                           hidden_layer_size=testing_parameter,
                                           data_size=arguments['DataSize'],
                                           training_percentage=arguments['TrainingPercentage'])
                train_success_rates.append(
                    mlp.train(epochs=arguments['Iterations'], learning_rate=arguments['LearningRate']))
                test_success_rates.append(mlp.test())
        elif test_type == TestType.LEARNING_RATE.value:
            testing_parameters.clear()
            for i in range(1, 11, 1):
                testing_parameters.append(i / 10)
                mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=arguments['HiddenLayers'],
                                           hidden_layer_size=arguments['HiddenLayerSize'],
                                           data_size=arguments['DataSize'],
                                           training_percentage=arguments['TrainingPercentage'])
                train_success_rates.append(mlp.train(epochs=arguments['Iterations'], learning_rate=i / 10))
                test_success_rates.append(mlp.test())
        else:
            for testing_parameter in testing_parameters:
                mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=arguments['HiddenLayers'],
                                           hidden_layer_size=arguments['HiddenLayerSize'],
                                           data_size=arguments['DataSize'],
                                           training_percentage=arguments['TrainingPercentage'])
                train_success_rates.append(mlp.train(epochs=testing_parameter, learning_rate=arguments['LearningRate']))
                test_success_rates.append(mlp.test())
        plt.title("Performance tests for training and testing set of " + test_type)
        tr_graph, = plt.plot(testing_parameters, train_success_rates, label='Training set')
        te_graph, = plt.plot(testing_parameters, test_success_rates, label='Test set')
        plt.legend(handles=[tr_graph, te_graph])
        plt.ylabel("Success Rates")
        plt.xlabel(test_type)
        plt.show()

    else:
        mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=arguments['HiddenLayers'],
                                   hidden_layer_size=arguments['HiddenLayerSize'], data_size=arguments['DataSize'],
                                   training_percentage=arguments['TrainingPercentage'])
        mlp.train(epochs=arguments['Iterations'], learning_rate=arguments['LearningRate'])
        mlp.test()
