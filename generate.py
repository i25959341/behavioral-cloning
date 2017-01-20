from scipy.misc import imread,imshow
import csv
import numpy as np
import random
from PIL import Image
from itertools import islice

OFFSET = 0.08
TOTALDATA=24107

# helper function for shuffling the data
def shuffle(data, steering):
    c = list(zip(data, steering))
    random.shuffle(c)
    data, steering = zip(*c)
    return data, steering

# helper function for resizing the image and normiliasation
def prepareImage(image):
    img = imread(image)
    img = np.array(img).astype(np.float32)
    img=img[60:134:2, 0:320:2,:]
    img = img/255 -0.5
    return img
# helper function for prpearing the label of the data
def prepare_label(steering):
    return np.array([float(steering)])

# read the csv file and get the img loction with its steerings angle
# also added offset to the left and right images for extra data
def read(filepath, folderName="data/"):
    with open(filepath) as csv_file:
        csvreader = csv.reader(csv_file, skipinitialspace=True)
        next(csvreader) # skip header row
        data =[]
        steerings=[]
        for row in csvreader:
            steering=prepare_label(float(row[3]))
            centre=(folderName+row[0], steering)
            left=(folderName+row[1], steering+OFFSET)
            right=(folderName+row[2], steering-OFFSET)

            data.append((centre[0],False))
            data.append((left[0],False))
            data.append((right[0],False))

            steerings.append(centre[1])
            steerings.append(left[1])
            steerings.append(right[1])

        return data, steerings

# helper fucntion for fliping the data along the vertical axis for extra data
def addFlipData(train, yTrain,pct):
    newTrain=[]
    newyTrain=[]
    for i in range(len(train)):
        newTrain.append(train[i])
        newyTrain.append(yTrain[i])

    num= int(pct*len(train))

    for i in range(num):
        steering = yTrain[i]
        nameBool = train[i][0]
        newTrain.append((nameBool,True))
        newyTrain.append(-steering)
    return newTrain, newyTrain

# Generator function for reading the data and getting the images into memeory
def generateBatch(names, y_data, batch_size = 32):
    total = len(names)
    current = 0
    while (True):
        imageData = np.zeros((batch_size, 37, 160, 3),dtype=float)
        steeringData = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            imageName=names[current][0]
            img = prepareImage(imageName)

            if names[current][1]==True:
                img=np.fliplr(img)
            imageData[j]= img

            steeringData[j]=y_data[current]
            current=(current+1)%total
        yield imageData, steeringData

# helper function for spliting the data into vlidaiton andt rainning
def splitData(data, steerings,pct):
    length=len(data)
    line = int(pct*length)
    return data[:line], steerings[:line], data[line:],steerings[line:]

# gneerator function for reading the csv file, fliping the data, getting the labals
# shuffling the data and yield the batch of validaiton and trainning data
def generate(filepath, pct, batchSize=32,flip=False):
    data, steerings = read("data/driving_log.csv", folderName="data/")
    data1, steerings1 = read("recovery/driving_log.csv", folderName="recovery/")
    data+=data1
    steerings+=steerings1
    data2, steerings2 = read("recovery1/driving_log.csv", folderName="recovery1/")
    data+=data2
    steerings+=steerings2
    data3, steerings3 = read("recovery2/driving_log.csv", folderName="recovery2/")
    data+=data3
    steerings+=steerings3

    data, steerings= shuffle(data,steerings)
    train, yTrain, valid, yValid = splitData(data, steerings, pct)
    if flip==True:
        train, yTrain = addFlipData(train, yTrain, 0.5)
    train, yTrain = shuffle(train, yTrain)
    return (generateBatch(train, yTrain),
            generateBatch(valid, yValid))

if __name__ == "__main__":
    train, valid = generate("data/driving_log.csv", 0.9,flip=True)
    t=next(train)
    v=next(valid)
    print(t[0].shape)
    # print(t[1])
