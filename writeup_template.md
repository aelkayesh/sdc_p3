**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/histogram.png "histogram"
[image2]: ./images/histogram1.png "histogram"
[image3]: ./images/arch.png "Model architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 12 (model.py lines 63-78) 

The model includes RELU layers to introduce nonlinearity (code line 66,68), and the data is normalized in the model using a Keras lambda layer (code line 65).  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The code also uses Max pooling layers of size 2x2. (model.py lines 67,69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 137 -150). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 144).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to apply some preprocessing to input images from the simulator and feed it to the model to learn from it.

My first step was to use a convolution neural network model. I used convolution layers along with activation and dropout layers. Finally fully connected layers are used to eventually reach a single output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I saw that the the model is doing well on training and validation sets so I thought it was ok for that step.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I noticed that most of the images have angles in the range of -0.165 to 0.1. For these images, I added only one out of every four, to make the input contain enought and fair training images for all angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 63-68) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



####3. Creation of the Training Set & Training Process

To train the model I used the data provided by Udacity. I tried adding extra data of my own, but after several times training the model, I figured out it was enough to use Udacity data.

The data was normalized using a lambda layer in the model. Before that, the image was cropped to remove the sky and the car hood, as they would just add noise and increase number of needed parameters.

To augment the data sat, I also flipped images and angles thinking that this would increase the number of training data I have, also, it would make the model drive good in roads having right or left curves, since the data was recorded using left curved tracks. I also added random brightness change to the images (line 100). Also, I changed color space to YUV, and I saw the model perform better in this color space. A final thing I did was resizing the image to 128 to 64, this helped me reduce the number of parameters by one third of what it was when using full image.



I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 . I used an adam optimizer so that manually training the learning rate wasn't necessary.
