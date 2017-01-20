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


# this model is baed on the NAIDIA papaer with several amendedations
# padding changed to same to accomodate smaller input size
# less feature maps and filters due to smaller input size
# used droped out the last layers for regularisations
def navidia():
    # number of convolutional filters to use
    nb_filters1 = 12
    nb_filters2 = 18
    nb_filters3 = 24
    nb_filters4 = 36

    # strides used
    stride1 = [1,1]
    stride2= [2,2]

    # convolution kernel size
    kernel_size = (3, 3)
    kernel_size1 = (3, 3)

    # Initiating the model
    model = Sequential()

    #Normialisation
    model.add(BatchNormalization(input_shape=(37, 160, 3)))
    # The first layer will turn 1 channel into 12 channels
    model.add(Convolution2D(nb_filters1, 3, 3, subsample=(1, 1), border_mode="same"))
    # Applying ReLU
    model.add(Activation('relu'))
    # The second conv layer will convert 12 channels into 18 channels
    model.add(Convolution2D(nb_filters2, 3, 3, subsample=(2, 2), border_mode="same"))
    # Applying ReLU
    model.add(Activation('relu'))
    # The  conv layer will convert 18 channels into 24 channels
    model.add(Convolution2D(nb_filters3, 3, 3, subsample=(2, 2), border_mode="same"))
    # Applying ReLU
    model.add(Activation('relu'))
    # The  conv layer will convert 24 channels into 36 channels
    model.add(Convolution2D(nb_filters4, 3, 3, subsample=(2, 2), border_mode="same"))
    # Applying ReLU
    model.add(Activation('relu'))
    # Apply dropout of 50%
    model.add(Dropout(0.5))
    # flatten the lyaers
    model.add(Flatten())

    model.add(Dense(500))
    # Applying ReLU
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100))
    # Applying ReLU
    model.add(Activation('relu'))

    model.add(Dense(50))
    # Applying ReLU
    model.add(Activation('relu'))
    # Apply dropout of 50%
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.summary()

    return model
