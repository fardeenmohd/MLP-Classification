import numpy as np
import pandas as pd
from enum import Enum

DATA_PATH_WHITE = "../resc/wine_quality/wine_quality_white.csv"
DATA_PATH_RED = "../resc/wine_quality/wine_quality_red.csv"


class WineType(Enum):
    WHITE = 1
    RED = 2


class WineQualityData:

    def __init__(self, number_of_inputs: int = 2000, train_percentage: int = 0.8, wine_type: WineType = WineType.WHITE):
        self.wine_type = wine_type
        self.data = pd.read_csv(DATA_PATH_RED, sep=';') if wine_type == WineType.RED else pd.read_csv(DATA_PATH_WHITE,
                                                                                                      sep=';')
        self.values = self.data.values
        self.number_of_rows = self.values.shape[0]
        self.number_of_columns = self.values.shape[1]
        self.number_of_inputs = number_of_inputs if number_of_inputs <= self.number_of_rows else self.number_of_rows
        self.number_of_rows = self.number_of_inputs
        self.train_percentage = train_percentage
        self.test_percentage = round(1 - train_percentage, 2)
        self.train_count = int(self.train_percentage * self.number_of_rows)
        self.test_count = int(self.test_percentage * self.number_of_rows)
        self.train_values = self.values[:self.train_count, :]
        self.test_values = self.values[:self.test_count, :]

        print("-----Parsed " + self.wine_type.name + " Wine Quality Data-----")
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

