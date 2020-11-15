# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is it! and here is a link to my [project code](https://github.com/KhardanOne/Traffic-Sign-Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the distribution of signs across the three sets.

![alt text][image1] TODO

The image shows that the distribution among the three sets is quite similar to each other.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After achieving barely 93% validation accuracy I started experimenting with ideas from the attached Yann LeCun paper.
As a first step, I converted the images to grayscale.

Here is an example of a traffic sign after grayscaling.

![alt text][image2] TODO

As a last step, I normalized the image.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer          	     	|     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input          	     	| 32x32x1 Grayscale image   							             | 
| Convolution 3x3      	| 1x1 stride, valid padding, outputs 28x28x9   	|
| RELU					             |	                                   											|
| Max pooling	      	   | 2x2 kernel, 2x2 stride, outputs 14x14x9     		|
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 14x14x24   |
| RELU                  |                                               |
| Max pooling           | 2x2 kernel, 2x2 stride, outputs 5x5x24        |
| Convert to fully conn.| outputs 600                                   |
| Fully connected		     | output 200                           									|
| RELU                  |                                               |
| Fully connected		     | output 80                            									|
| RELU                  |                                               |
| Fully connected		     | output 43                            									|
| Softmax			          	 |                                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Before training I filled the weights and biases in the model with random_normal numbers, with sigma = 0.1 and mu = 0 values. 

I used the Adam optimizer with a learning rate of 0.001.
I measured the cross entropy between the logits and the expected outcome, and took the mean values of CEs across the batch to feed the optimizers minimize() function.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95% 
* test set accuracy of ? TODO *************************************************************************

This is a result of an iterative process.
* First I did not specify the sigma value for generating inital variables. That resulted in 0% accuracy. So I chose a sigma value of 0.1
* In the inital setup all layers were 2 times wider.
* The training accuracy almost reached 93%.
* I read the Yann LeCun whitepaper, and for starter I decided to convert my images to grayscale.
* I also halved the withs to their actual values. The consideration was that grayscale images contain less data than color images, requiring less nodes.
* This resulted in 95% validation accuracy.
* Then I lowered the number of epochs to one temporarily to lessen the time needed for other experimentations.
* My own images scored 80% and 100%s.
* Just before handing in the solution I turned ON the GPU, and surprise, surprise, those numbers dropped to 0%. Turning it off again my results were good again. It seems that there is something strange going on with that Udacity Workspace GPUs.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Road narrows on the right   									| 
| U-turn     			| Go straight or left  										|
| Yield					| Wild animals crossing   							|
| 100 km/h	      		| Priority road			 				|
| Slippery Road			| Bumpy road        							|

TODO

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


