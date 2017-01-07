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

def navidia():
    nb_filters1 = 12
    nb_filters2 = 18
    nb_filters3 = 24
    nb_filters4 = 36
    nb_filters5 = 36

    pool_size = (2, 2)

    stride1 = [1,1]
    stride2= [2,2]

    kernel_size = (3, 3)
    kernel_size1 = (3, 3)

    model = Sequential()

    model.add(BatchNormalization(input_shape=(37, 160, 3)))

    model.add(Convolution2D(nb_filters1, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters2, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters3, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters4, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.summary()

    return model


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
