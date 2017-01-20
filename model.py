from generate import generate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from modelSetting import modelSetting, navidia
import numpy as np
import json
import math
from keras.optimizers import Adam, RMSprop

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Paramters on number of peochs , batchsize and total number of datas, pect of
# training vs validation data
nb_epochs=10
batchSize=128
total_examples = 38751
pct_train = 0.9

# number of training and number of vlidation data, no testing required
nb_train= int(total_examples*pct_train)
nb_valid = total_examples-nb_train

# generator for the training and validation data
train, valid = generate("data/driving_log.csv", 0.9, batchSize)

# Initiating the model Navidia model for trainning
model = navidia()

# Compile model using Adam optimizer
# and loss computed by mean squared error
model.compile(loss='mse', optimizer=Adam(lr=0.0001))
### Model training
model.fit_generator(generator=train,
                    samples_per_epoch=nb_train,
                    nb_epoch=nb_epochs,
                    nb_val_samples=nb_valid,
                    validation_data=valid,
                    callbacks=[checkpoint, callback])
# Save the model.
json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
print("Model Saved")
