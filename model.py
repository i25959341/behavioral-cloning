from generate import generate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from modelSetting import modelSetting
import numpy as np
import json
import math
from keras.optimizers import Adam, RMSprop

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

nb_epochs=20
batchSize=128
total_examples = 24107

pct_train = 0.9

nb_train= int(total_examples*pct_train)
nb_valid = total_examples-nb_train

train, valid = generate("data/driving_log.csv", 0.9, batchSize)

model = modelSetting()

model.compile(loss='mse', optimizer=Adam(lr=0.0001))

model.fit_generator(generator=train,
                    samples_per_epoch=nb_train,
                    nb_epoch=nb_epochs,
                    nb_val_samples=nb_valid,
                    validation_data=valid,
                    callbacks=[checkpoint, callback])

json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
print("Model Saved")
