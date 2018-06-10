from enum import Enum
import pandas as pd
from sklearn import preprocessing

DATA_PATH_TEST_POKER = "../resc/poker_hands/poker_hands_test.data"
DATA_PATH_TRAIN_POKER = "../resc/poker_hands/poker_hands_train.data"

DATA_PATH_WHITE_WINE = "../resc/wine_quality/wine_quality_white.csv"
DATA_PATH_RED_WINE = "../resc/wine_quality/wine_quality_red.csv"
DATA_PATH_DRUG = "../resc/drug_consumption/drug_consumption.data"
DRUG_DATA_PERSONALITY_END = 12


class DataType(Enum):
    WHITE_WINE = 'WhiteWine'
    RED_WINE = 'RedWine'
    DRUG_CONSUMPTION = 'DrugConsumption'
    POKER_HANDS = 'PokerHands'


def acquire_data(data_type: DataType, number_of_inputs: int):
    if data_type == DataType.DRUG_CONSUMPTION:
        data = pd.read_csv(DATA_PATH_DRUG)
        number_of_rows = number_of_inputs if number_of_inputs <= data.shape[0] else data.shape[0]
        data = data.values[:number_of_rows, 1:]  # to remove the first column which is just the ID number
        personality_data = data[:, DRUG_DATA_PERSONALITY_END:]
        for x in range(len(personality_data)):
            for y in range(len(personality_data[0])):
                personality_data[x][y] = float(personality_data[x][y].strip()[2])
        data[:, DRUG_DATA_PERSONALITY_END:] = personality_data
        return data
    elif data_type == DataType.WHITE_WINE:
        data = pd.read_csv(DATA_PATH_WHITE_WINE, sep=';')
        number_of_rows = number_of_inputs if number_of_inputs <= data.shape[0] else data.shape[0]
        return data.values[:number_of_rows, :]  # to remove the first column which is just the ID number
    elif data_type == DataType.RED_WINE:
        data = pd.read_csv(DATA_PATH_RED_WINE, sep=';')
        number_of_rows = number_of_inputs if number_of_inputs <= data.shape[0] else data.shape[0]
        return data.values[:number_of_rows, :]  # to remove the first column which is just the ID number
    else:
        data = (pd.read_csv(DATA_PATH_TRAIN_POKER).values, pd.read_csv(DATA_PATH_TEST_POKER).values)
        total_data_rows = data[0].shape[0] + data[1].shape[0]
        number_of_rows = number_of_inputs if number_of_inputs <= total_data_rows else total_data_rows
        data = (pd.read_csv(DATA_PATH_TRAIN_POKER).values[:number_of_rows, :],
                pd.read_csv(DATA_PATH_TEST_POKER).values[:number_of_rows, :])

        return data


class DataClass:

    def __init__(self, data_type: DataType = DataType.POKER_HANDS, number_of_inputs: int = 1500,
                 training_percentage: float = 0.8):
        self.type = data_type
        self.number_of_inputs = number_of_inputs
        self.data = acquire_data(data_type, self.number_of_inputs)
        self.shape = self.data[0].shape if data_type == DataType.POKER_HANDS else self.data.shape
        self.number_of_rows = self.shape[0]
        self.number_of_columns = self.shape[1]
        self.training_percentage = training_percentage
        self.test_percentage = round(1 - self.training_percentage, 2)
        self.train_row_count = int(self.training_percentage * self.number_of_rows)
        self.test_row_count = int(self.test_percentage * self.number_of_rows)
        self.train_values = self.data[0][:self.train_row_count, :] if data_type == DataType.POKER_HANDS else \
            self.data[:self.train_row_count, :]
        self.test_values = self.data[0][:self.test_row_count, :] if data_type == DataType.POKER_HANDS else \
            self.data[:self.test_row_count, :]

        # print("-----Parsed " + self.type.value.__str__() + " Data-----")
        # print("Total input rows demanded: " + number_of_inputs.__str__() + " Total input rows parsed: " +
        #       (self.train_row_count + self.test_row_count).__str__())
        # print("Test count: " + (self.test_values.shape[0]).__str__() + " Train count: " +
        #       (self.train_values.shape[0]).__str__())
        # print("Number of Attributes: " + self.number_of_columns.__str__())

        self.test_x, self.test_y, self.train_x, self.train_y = self.split_x_y_sets()

        self.train_x = preprocessing.normalize(self.train_x)
        self.test_x = preprocessing.normalize(self.test_x)

        # print("Train X has " + self.train_x.shape[0].__str__() + " rows and "
        #       + self.train_x.shape[1].__str__() + " columns.")
        # print("Train Y has " + self.train_y.shape[0].__str__() + " rows and "
        #       + self.train_y.shape[1].__str__() + " columns.")
        # print("Test X has " + self.test_x.shape[0].__str__() + " rows and "
        #       + self.test_x.shape[1].__str__() + " columns.")
        # print("Test Y has " + self.test_y.shape[0].__str__() + " rows and "
        #       + self.test_y.shape[1].__str__() + " columns.")
        self.features_x = self.train_x.shape[1]
        self.features_y = self.train_y.shape[1]

    def split_x_y_sets(self):
        if self.type == DataType.DRUG_CONSUMPTION:

            return (self.test_values[:, :DRUG_DATA_PERSONALITY_END],
                    self.test_values[:, DRUG_DATA_PERSONALITY_END:],
                    self.train_values[:, :DRUG_DATA_PERSONALITY_END],
                    self.train_values[:, DRUG_DATA_PERSONALITY_END:])
        else:
            return (self.test_values[:, : - 1],
                    self.test_values[:, - 1:].reshape(-1, 1),
                    self.train_values[:, : - 1],
                    self.train_values[:, - 1:].reshape(-1, 1))
