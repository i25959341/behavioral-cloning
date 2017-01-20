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
The goal is to drive a car autonomously in a simulator using a deep neuronal network (DNN) trained on human driving behavior. For that Udacity provided the simulator and a basic python script to connect a DNN with it. The Network is only passed images from a front facing camera and the normalized steering angle in which the vehicle is being turned.

## Input Data gathering
Collecting the data was one of the most important steps during this project. During trainning, the simulator records three images with a frequency of 10hz. Next to a camera centered at the car there are also two additional cameras recording with an offset to the left and right respectively. This allows to apply an approach described in a paper by Nvidia. 

## Data preparation
I found that the whole image can confuse the model due to unncessary background noises such as tries, skies, etc. I decided to cut those unncessary pixels and reduced the size by 25%. As a result, the size of the image was 37 x 160 x 3. The model was trained on center camera, however, it was oscillating quite a lot in the middle of the track and there was very poor recovery. I decided to add also left and right image, with the angle adjustment of 0.1. This parameter was tuned by trial and error.

## Model Architecture Design

The model architecture is basiclly the same as the Navidia paper shown by Udacity, but with severial important modifications. Firstly, number of filters are havled as our image input size is much smaller than Navidia. Second, the model used dropout to improved reularisations; it is noticeable that once regulisation is added, the car swings less and more stable.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
batchnormalization_1 (BatchNorma (None, 37, 160, 3)    12          batchnormalization_input_1[0][0] 
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 37, 160, 12)   336         batchnormalization_1[0][0]       
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 37, 160, 12)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 19, 80, 18)    1962        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 19, 80, 18)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 40, 24)    3912        activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10, 40, 24)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 20, 36)     7812        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 5, 20, 36)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5, 20, 36)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3600)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           1800500     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 500)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 500)           0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        activation_6[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           activation_7[0][0]               
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             51          dropout_3[0][0]                  
____________________________________________________________________________________________________
```

## Data Processing
Udacity data is used as the foundation of the project but it is shown that it is not enough to get good results. This might be due to Udacity dataset is quite limited and there is a lot of zero-centred data. There is a tendency of the car not moving at all due to most of the time in the trainning data, the car dont move and thus influence its behavior in curves.

Therefore, I have used a joystick to record more data with a focus on colllecting data along curves. I avoid recording data in the straight road only when I am steering. Several difficult part of the roads are recorded repeatedly to get good behaviour. I also focus recoding data on the part of the road where the car made mistakes which are detailed in the next sections.

Data collection singlehandly improved most of the performance of the car very significantly

## Track - The Bridge, The 1st Corner after the bridge with dirt, the 2nd Coarn after the bridge turning right
After testing on autonomous mode, it became obvious that the car was struggling with several part of the track significant and there were the bridge, the 1st corner after the bridge and the 2nd right corner after the bridge.

The bridge was difficult as it has a very different colour scheme versus the rest of the road. It was difficult for the car to recognise the edge of the road and often run into the wall.

The 1st corner was tricky as not only the turn was very sharp but it also has a dirt right in front which can confuse the car thinking that it is a straight road.

The right turn was interesting as it shows the importance of not overfitting. We have only trainned the car to turn left at this point since most of the track was left turned. In addition, this turn also has a massive blue sky in front that can also confuse the neural network

All these difficulties were fixed by adding additional data to these specific corners. Recovery data are added whenever the model failed to performed a specific manevevre when needed.

## Conclusion
The most important thing is Data!


Video link to the recording
http://youtu.be/84MMQiyglVw?hd=1



