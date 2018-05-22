import numpy as np
import pandas as pd

DATA_PATH_TEST = "../resc/poker_hands/poker_hands_test.data"
DATA_PATH_TRAIN = "../resc/poker_hands/poker_hands_train.data"


class PokerHandsData:

    def __init__(self, number_of_inputs: int = 2000, train_percentage: int = 0.8):
        self.train_data = pd.read_csv(DATA_PATH_TEST)
        self.test_data = pd.read_csv(DATA_PATH_TRAIN)
        self.train_values = self.train_data.values
        self.test_values = self.test_data.values
        self.total_data_rows = self.train_data.values.shape[0] + self.test_data.values.shape[0]
        self.number_of_rows = number_of_inputs if number_of_inputs <= self.total_data_rows else self.total_data_rows
        self.number_of_columns = self.test_values.shape[1]
        self.train_percentage = train_percentage
        self.test_percentage = round(1 - train_percentage, 2)
        self.train_count = int(self.train_percentage * self.number_of_rows)
        self.test_count = int(self.test_percentage * self.number_of_rows)
        self.train_values = self.train_values[:self.train_count, :]
        self.test_values = self.test_values[:self.test_count, :]

        print("-----Parsed Poker Hands Data-----")
        print("Total input rows demanded: " + number_of_inputs.__str__() + " Total input rows parsed: " +
              (self.train_count + self.test_count).__str__())
        print("Test count: " + (self.test_values.shape[0]).__str__() + " Train count: " +
              (self.train_values.shape[0]).__str__())
        print("Number of Attributes: " + self.number_of_columns.__str__())

        self.train_x = self.train_values[:, : - 1]
        self.train_y = self.train_values[:, - 1:].reshape(-1, 1)
        self.test_x = self.test_values[:, : - 1]
        self.test_y = self.test_values[:, - 1:].reshape(-1, 1)

        print("Test X has " + self.test_x.shape[1].__str__() + " attributes")
        print("Test Y has " + self.test_y.shape[1].__str__() + " attributes")
        print("Train X has " + self.train_x.shape[1].__str__() + " attributes")
        print("Train Y has " + self.train_y.shape[1].__str__() + " attributes")