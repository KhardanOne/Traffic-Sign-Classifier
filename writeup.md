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

[image1]: ./doc/frequency.png "Sign Frequency"
[image2]: ./doc/grayscale.png "Grayscaling"
[image4]: ./webimages/1.png "Traffic Sign 1"
[image5]: ./webimages/2.png "Traffic Sign 2"
[image6]: ./webimages/3.png "Traffic Sign 3"
[image7]: ./webimages/4.png "Traffic Sign 4"
[image8]: ./webimages/5.png "Traffic Sign 5"
[image9]: ./doc/certainty.png "Certainty"

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

![Sign Frequency][image1]

The image shows that the distribution among the three sets is quite similar to each other.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After achieving barely 93% validation accuracy I started experimenting with ideas from the attached Yann LeCun paper.
As a first step, I converted the images to grayscale.

Here is an example of a traffic sign after grayscaling.

![Grayscaling][image2]

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
I measured the cross entropy between the logits and the expected outcome, and took the mean values of CEs across the batch to feed the optimizers minimize() function. Batch size was 128 and I used 60 epochs.

Analyzing the training and validation accuracies especially the differencies between the training and validation accuracies show that the model is overtraining. Experimenting further with different kind of regularizations method could further improve the validation and test accuracy.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100.0%
* validation set accuracy of 95.2% 
* test set accuracy of 93.7%

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

The images are of relatively good quality. Difficulties in classification might arise from shadows (image 1), over-exposure (3), blue fringing (3) and under-exposure. I did normalize the images so that the exposure problems are lessened to a manageable level.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road narrows on the right      		| Road narrows on the right   									| 
| Go straight or left    			| Go straight or left  										|
| Wild animals crossing				| Wild animals crossing   							|
| Priority road	      		| Priority road			 				|
| Bumpy road  			| Bumpy road        							|

The model was able to correctly guess all the 5 traffic signs, which gives an accuracy of 100%. Of course this 100% is only possible because of the small number of signs. Accuracy would converge to the test accuracy, had we have more signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The below image shows for all 5 signs the softmax probabilities for the top 5 values each. Signs correspond to rows, probabilities to columns. Note that softmaxing 43 classes caused the otherwise big differences between logits become much smaller. Reason: the softmax method allows for 1 percentages even for logits of zero values, exhausting the available 100 percents fast. Looking at logits (not displayed here) we can see that the prediction was quite confident in its choices. Much more confident than the below images suggest.

![Certainty][image9]



