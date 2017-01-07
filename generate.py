from scipy.misc import imread,imshow
import csv
import numpy as np
import random
from PIL import Image
from itertools import islice

OFFSET = 0.08
TOTALDATA=24107

def shuffle(data, steering):
    c = list(zip(data, steering))
    random.shuffle(c)
    data, steering = zip(*c)
    return data, steering

def prepareImage(image):
    path = "data/"+image
    img = imread(path)
    img = np.array(img).astype(np.float32)
    img=img[60:134:2, 0:320:2,:]
    img = img/255 -0.5
    return img

def prepare_label(steering):
    return np.array([float(steering)])

def read(filepath):
    with open(filepath) as csv_file:
        csvreader = csv.reader(csv_file, skipinitialspace=True)
        next(csvreader) # skip header row
        # data= [(row[0], row[1], row[2], row[3]) for row in csvreader]
        data =[]
        steerings=[]
        for row in csvreader:
            steering=prepare_label(float(row[3]))
            centre=(row[0], steering)
            left=(row[1], steering+OFFSET)
            right=(row[2], steering-OFFSET)

            data.append((centre[0],False))
            data.append((left[0],False))
            data.append((right[0],False))

            steerings.append(centre[1])
            steerings.append(left[1])
            steerings.append(right[1])

        return data, steerings

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

def splitData(data, steerings,pct):
    length=len(data)
    line = int(pct*length)
    return data[:line], steerings[:line], data[line:],steerings[line:]

def generate(filepath, pct, batchSize=32,flip=False):
    data, steerings = read(filepath)
    data, steerings= shuffle(data,steerings)
    train, yTrain, valid, yValid = splitData(data, steerings, pct)
    if flip==True:
        train, yTrain = addFlipData(train, yTrain,0.5)
    train, yTrain = shuffle(train, yTrain)
    print (len(data))
    print (len(train))
    print (len(valid))
    return (generateBatch(train, yTrain),
            generateBatch(valid, yValid))

# prepareImage("IMG/center_2016_12_01_13_33_13_214.jpg")
if __name__ == "__main__":
    train, valid = generate("data/driving_log.csv", 0.9,flip=True)
    t=next(train)
    v=next(valid)
    print(t[0].shape)
    # print(t[1])
