from data.data_class import DataClass, DataType
from src.mlp import MultiLayerPerceptron

if __name__ == "__main__":
    poker_mlp = MultiLayerPerceptron(data_type=DataType.POKER_HANDS, data_size=1500)
    poker_mlp.train()
    #poker_mlp.test()
