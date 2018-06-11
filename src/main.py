from argparse import ArgumentParser

from data.data_class import DataType
from src.mlp import MultiLayerPerceptron
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-d', '--DataType', default=DataType.POKER_HANDS.value, type=str,
                        help='WhiteWine, RedWine, DrugConsumption or PokerHands')
    parser.add_argument('-ds', '--DataSize', default=1500, type=int, help='Size of data')
    parser.add_argument('-l', '--HiddenLayers', default=2, type=int, help='Number of hidden layers')
    parser.add_argument('-ls', '--HiddenLayerSize', default=20, type=int, help='Size of the hidden layer')
    parser.add_argument('-tp', '--TrainingPercentage', default=0.8, type=int, help='Training Percentage')
    parser.add_argument('-i', '--Iterations', default=10, type=int, help='Number of epochs')
    parser.add_argument('-lr', '--LearningRate', default=0.5, type=float, help='Learning rate')
    parser.add_argument('-t', '--Test', default=False, type=bool,
                        help='Add this argument if you want to test hidden layer size')
    parser.add_argument('-str', '--Start', type=int, default=100, help='Starting point for testing')
    parser.add_argument('-stp', '--Stop', type=int, default=500, help='Stopping point for testing')
    parser.add_argument('-st', '--Step', type=int, default=100, help='Steps for testing')

    arguments = vars(parser.parse_args())
    print(arguments)
    parser.print_help()

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
        sizes = []
        errors = []

        start = arguments['Start']
        stop = arguments['Stop']
        step = arguments['Step']

        for i in range(start, stop, step):
            sizes.append(i)

        for size in sizes:
            mlp = MultiLayerPerceptron(data_type=data_type, hidden_layer_size=size, data_size=5000)
            mlp.train()
            errors.append(mlp.test())

        plt.plot(sizes, errors)
        plt.ylabel("Errors")
        plt.xlabel("Hidden Layer Size")
        plt.show()

    else:
        mlp = MultiLayerPerceptron(data_type=data_type, number_of_hidden_layers=arguments['HiddenLayers'],
                                   hidden_layer_size=arguments['HiddenLayerSize'], data_size=arguments['DataSize'],
                                   training_percentage=arguments['TrainingPercentage'])
        mlp.train(epochs=arguments['Iterations'], learning_rate=arguments['LearningRate'])
        #mlp.test()
