from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization

def modelSetting():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(37, 160, 3)))

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))

    model.add(MaxPooling2D())

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.summary()

    return model
