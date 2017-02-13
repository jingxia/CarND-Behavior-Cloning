


# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: nvidia.png "Nvidia model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
 python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I started with the LeNet structure but later switched to NVidia's model for better performance. My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64. Code snippet below:
```sh
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
```

The model includes RELU layers to introduce nonlinearity, and the data is normalized using a Keras lambda layer. Cropping is also introduced to shift more focus on the road itself(top 70 and bottom 25 pixels are cropped).

```
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

```
X_train, X_validation, y_train, y_validation = train_test_split(X_sample, y_sample, test_size=0.2)

```

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```sh
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet, which did not work too well for me and the car was driving straight off the track. I switchted to the Nvidia model later on and was able to achieve a better result.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. Three 5*5 layers with subsampling. and two 3* 3 layers. I played around with adding dropout layers as well. However, it did not seem to gain much improvement over the original nvidia model. Thus I kept the Nvidia model as is.

Here is a visualization of the architecture(Nvidia model)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Initially I used the old simulator to drive around the track but was not able to gain good results. So I switched to use dataset provided by Udacity for more accurate data points. I added additional images by flipping the data and also added recovery data by adding left and right camera images with corrected angles.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 and I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. driving video

[![Driving video](https://youtu.be/EHlRP4ycjZM)](https://youtu.be/EHlRP4ycjZM "Driving Video")







```python

```
