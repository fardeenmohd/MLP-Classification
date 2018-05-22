import numpy as np
import pandas as pd
import math

DATA_PATH = "../resc/drug_consumption/drug_consumption.data"


class DrugConsumptionData:

    def __init__(self, number_of_inputs: int = 1500, train_percentage: float = 0.8):
        self.data = pd.read_csv(DATA_PATH)
        self.values = self.data.values[:, 1:]  # to remove the first column which is just the ID number
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
        print("-----Parsed Drug Consumption Data-----")
        print("Total input rows demanded: " + number_of_inputs.__str__() + " Total input rows parsed: " +
              (self.train_count + self.test_count).__str__())
        print("Test count: " + (self.test_values.shape[0]).__str__() + " Train count: " +
              (self.train_values.shape[0]).__str__())
        print("Number of Attributes: " + self.number_of_columns.__str__())