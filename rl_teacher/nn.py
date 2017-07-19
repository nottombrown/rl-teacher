from keras.layers import Dense, Dropout, LeakyReLU

def two_layer_fc_net(x, h_size=64):
    """Vanilla two hidden layer multi-layer perceptron"""
    x = Dense(h_size)(x)
    x = LeakyReLU()(x)

    x = Dropout(0.5)(x)
    x = Dense(h_size)(x)
    x = LeakyReLU()(x)

    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    return x
