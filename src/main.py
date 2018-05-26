from data.data_class import DataClass, DataType

if __name__ == "__main__":
    poker_data_class = DataClass(DataType.POKER_HANDS, 5)
    drug_data_class = DataClass(DataType.DRUG_CONSUMPTION, 5)
    white_wine_data_class = DataClass(DataType.WHITE_WINE, 5)
    red_wine_data_class = DataClass(DataType.RED_WINE, 5)
