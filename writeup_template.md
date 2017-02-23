#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

For this project, I implemented the model architecture laid out in NVIDIA's End to End Learning for Self-Driving Cars paper. The model consists of an input layer with cropping (model.py line 34). The cropping served to reduce the input size and helped to train the model on the primary area of interest in the camera view of the road. A lambda layer was used to scale the input data to a range of -0.5 to 0.5 to prevent uncharacteristically high pixel values from having an artificially high corresponding activation likelihood. Next, the model has several 5x5 and 3x3 convolution layers. Valid padding was used to reduce the width of the network as we increased the depth from layer to layer. Following the convolution layers, the model was flattened and passed through several dense (fully connected) layers. The last layer consists of one output corresponding to the inferred steering angle given the state input to the model. One hot encoding is not used in this case, as the output is a classification probablity estimate. For each layer, a rectified linear unit (relu) layer was used for activation in order to introduce non-linearity into the model. Finally, one dropout layer was added (model.py line 57) in order to reduce the likelihood of over-fitting.

####2. Attempts to reduce overfitting in the model

As stated in part 1, the model contains a dropout layer in order to reduce overfitting (model.py line 57). 

The data was split, 80% for training and 20% for validation, in order to minimize overfitting (model.py line 16) and the resulting model was tested on track 1 over several laps.
####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68). I experimented with the number and associated probabilities of dropout layers. I settled on using a single dropout layer with a keep probability of 0.5 (model.py line 57)

####4. Appropriate training data

Only training data from track 1 was used. All 3 camera views were utilized in training. Details as follows.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to base the model on NVIDIA's work. Utilizing several consecutive convolution layers proves very effective in encoding low, mid, and high level image features in the model. By utilizing this approach, the model can be trained to infer features like edges, lines, and corresponding lane and road markings in a more robust manner when compared to traditional computer vision approaches as exampled in project one.


In creating my training and validation data, I decided to utilize all 3 camera views. The left and right camera views provided a good means for artificially creating a state of the vehicle traveling along the sides of the road. For the left and right camera images, the steering angle was corrected by a factor of +/-0.2 in order to create a state-action pair that would train the network to always drive towards the center of the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The training process went as expected with no obvious signs of overfitting. having trained the model, testing in the simulator was the next hurdle. 

Initially, the model performed very well. It seemed to have some trouble along the cobblestone bridge as that area of the track differed greatly and was underrepresented in my training data. The car was able to successfully navigate the bridge despite showing some small perturbations in steering angle.

The car failed to stay on the road at the very first right turn. It was instantly apparent that this state-action pair was significantly under-represented in the training data. In order to correct this, I simply extended the training data by flipping camera images and corresponding steering angles for left turn areas of the track, thereby artificially creating data to train right turns. Additionally, I increased the steering angle correction values for the left and right camera images in order to train a more aggressive lane centering behaviour.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 33-66) consisted of a convolution neural network. The layer types and shapes are shown clearly in the image below.

![model](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/model.png)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image_center](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/data/IMG/center_2016_12_01_13_32_45_477.jpg)

I also used images from the left side and right camera view. Here is a selection off all 3 views:

![image_left](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/data/IMG/left_2016_12_01_13_32_45_477.jpg)
![image_center](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/data/IMG/center_2016_12_01_13_32_45_477.jpg)
![image_right](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/data/IMG/right_2016_12_01_13_32_45_477.jpg)

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![image_center](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/original.jpg)
![image_flipped](/home/daniel/SDC/CarND-Behavioral-Cloning-P3/flipped.jpg)

After the collection process, I had 24,108 number of data points. I then doubled my data by flipping all images and corresponding steering angles.

I finally randomly shuffled the data set and put 20% of the data into a validation set using sklearn train_test_split function. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by plateau in loss at about epoch 4 or 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
