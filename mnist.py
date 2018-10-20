import keras
from keras.datasets import mnist
from keras.models import Sequential


def main():
    """
    Simple NN with keras
    :return:
    """
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
