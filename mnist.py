from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses, optimizers
from keras.utils import np_utils
import numpy as np


def simple_model(input_shape):

    # model structure
    model = Sequential()
    model.add(Dense(units=10, input_shape=input_shape))
    model.add(Activation('softmax'))
    # choose optimizer and compile
    opt = optimizers.SGD()
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=["accuracy"])
    # print model summary
    model.summary()

    return model


def main():
    """
    Simple NN with keras
    """
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    ntrain, img_width, img_height = X_train.shape
    ntest = X_test.shape[0]
    num_class = 10

    # data processing: important to reshape 2d images into 1d data
    # because in the simple model we just use dense layer, no 2dconv
    X_train = X_train.reshape(ntrain, img_width*img_height).astype(np.float32)
    X_test = X_test.reshape(ntest, img_width*img_height).astype(np.float32)
    # normalize data
    X_train /= 255.
    X_test /= 255.
    # convert labels to categorical
    y_train = np_utils.to_categorical(y_train, num_classes=num_class)
    y_test_binary = np_utils.to_categorical(y_test, num_classes=num_class)
    # training
    batch_size = 128
    epochs = 5

    model = simple_model(input_shape=(img_width*img_height, ))
    history = model.fit(X_train, y_train, batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)


    # predict classes
    y_predict = model.predict_classes(X_test)
    # print(y_predict.shape)
    # print(y_predict)

    # manually check predicted class by printing true class and predicted class side by side
    for y1, y2 in zip(y_test, y_predict):
        print("true class:%s, predicted class:%s"%(y1, y2))

    # evaluate the predictions using function evaluate
    score = model.evaluate(X_test, y_test_binary, verbose=0)
    # print loss value and accuracy on test data
    print("test loss: %s"%score[0])
    print("test accuracy: %s"%score[1])




if __name__ == "__main__":
    main()

