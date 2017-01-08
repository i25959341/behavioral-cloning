# behavioral-cloning

## Overview

There are 3 python files for this project

## generate.py
1. This python script imports the steering angles and respective filenames from the csv files
2. This data is used to perform image generations on a batch of N images.
3. The generations performs normalisation and cropp and resize the images
4. During the process, it also perform reflections on the images to get extra set of image with opposite steeering angles for data augmentation
5. This also perform data shuffling and train/valid data split. No test set is used, as it is not very useful. The best test is to drive!

## modelSetting.py
1. This file gives the archietature of the model that I am using for learning
2. The model is basiclly the one that Navidia uses with less number of filters in each conv layer as our images are smaller
3. Dropout are added for regularisation


## model.py
1. This file uses the tranning model from modelSetting.py and connect it to the generators
2. The file then performs trainnning and there are several high levels parameters to fiddle like learning-rate, epochs, bathc-size etc
3. Adam optimiser is used for the less parameter to worry about.
4. When the training is done, the model and weights are saved as model.json and model.h5.

## drive.py
1. This is the python script that receives the data from the Udacity program, predicts the steering angle using the deep learning model, and send the throttle and the predicted angles back to the program.
2. Since the images were reshaped and normalized during training, the image from the program is reshaped and normalized just as in generate.py and model.py


## Challenges

## Input Data gathering

## Track

## Model network architecture



## Data preparation


## Conclusion
