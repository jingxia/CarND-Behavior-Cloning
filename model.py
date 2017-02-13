
# coding: utf-8

# In[10]:

import matplotlib.image as mpimg
import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import os
import json
import csv

def generator(X_samples, y_samples, batch_size=128):
    num_samples = len(X_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(X_samples, y_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples_X = X_samples[offset:offset+batch_size]
            batch_samples_y = y_samples[offset:offset+batch_size]
            yield shuffle(batch_samples_X, batch_samples_y)


def get_model(time_len=1):
    """
    A slightly modified version of the Nvidia model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model



lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []

# Append left and right cameras to the originial images with angle correction(+-0.2)
for line in lines:
    for i in range(3):
        path = line[i]
        tokens = path.split('/')
        filename = tokens[-1]
        local_path = 'IMG/'+filename
        image = cv2.imread(local_path)
        images.append(image)
    correction = .2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    
aug_imgs = []
aug_msmt = []

# Append flipped images to the samples
for image, measurement in zip(images, measurements):
    aug_imgs.append(image)
    aug_msmt.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_msmt = measurement * -1.0
    aug_imgs.append(flipped_image)
    aug_msmt.append(flipped_msmt) 
    
X_sample = np.array(aug_imgs)
y_sample = np.array(aug_msmt)


X_train, X_validation, y_train, y_validation = train_test_split(X_sample, y_sample, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=128)
validation_generator = generator(X_validation, y_validation, batch_size=128)


model = get_model()
model.fit_generator(train_generator, samples_per_epoch=len(X_train), validation_data=validation_generator,
                    nb_val_samples=len(X_validation), nb_epoch=3)
model.save("model.h5")
print("Saving model weights and configuration file.")






# In[ ]:



