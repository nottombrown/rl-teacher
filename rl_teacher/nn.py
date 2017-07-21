from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential

class FullyConnectedMLP(object):
    def __init__(self, input_dim, h_size=64):
        """Vanilla two hidden layer multi-layer perceptron"""
        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, x):
        return self.model(x)
