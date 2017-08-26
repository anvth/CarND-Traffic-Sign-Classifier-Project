#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./reportImages/bar_chart.png "Visualization"
[image2]: ./newImages/bikecrossing.jpg "Bike Crossing"
[image3]: ./newImages/childrencrossing.jpg "Children Crossing"
[image4]: ./newImages/nopassing.jpg "No Passing"
[image5]: ./newImages/stop.jpg "Stop"
[image6]: ./newImages/straightorright.jpg "Straight or Right"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is (31367, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

| | ClassId | SignName         		|     Occurance	        					| 
|:-:|-------:|:--------------:|:---------------------------------------------:| 
|0|2|Speed limit (50km/h)|1799|
|1|1|Speed limit (30km/h)|1781|
|2|13|Yield|1730|
|3|12|Priority road|1691|
|4|38|Keep right|1627|
|5|10|No passing for vehicles over 3.5 metric tons|1619|
|6|4|Speed limit (70km/h)|1578|
|7|5|Speed limit (80km/h)|1485|
|8|9|No passing|1189|
|9|25|Road work|1181|

```
Min number of images per class = 161
Max number of images per class = 1799
```

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the training and test set. And in the second step, I normalized the data and converted it to 32-bit floating point. The data was not converted to grayscale.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My network is same as the one proposed in LeNet described in the course material. Except for the below updations:
1. The layer 1 would except depth-3 instad of the previous depth-1
2. Added a dropout after every relu layer(activation function)
3. Changed the final output classes to 43 from 10

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   					| 
| Layer 1: Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU			|								| 
| Dropout      		| tunable parameter 					|
| Max pooling	      | 2x2 stride,  outputs 14x14x6 			|
| Layer 2: Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU			|								| 
| Dropout      		| tunable parameter					|
| Max pooling	      | 2x2 stride,  outputs 5x5x16 			|
| Flatten	      	| Input 5x5x16, output 400 				|
| Layer 3: Fully connected		| Input 400, output 120				|
| RELU			|								| 
| Dropout      		| tunable parameter					|
| Layer 4: Fully connected		| Input 120, output 84					|
| RELU			|								| 
| Dropout      		| tunable parameter					|
| Layer 5: Fully connected		| Input 84, output 43 (labels)					|

The output of layer 5 better known as logits. I took the cross-entropy of logits with one-hot encoded labels. The loss was defined to be the average of the cross entropy across the batch. This computed loss was then supplied to the optimizer. 
 
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
EPOCHS = 40
BATCH_SIZE = 128
rate = 0.001
dropout = .50

Changed dropout to 1.0 for validation and testing

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7
* validation set accuracy of 98.9
* test set accuracy of 93.0

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bike Crossing      		| Bike Crossing   									| 
| Children Crossing     			| Children Crossing 										|
| No Passing					| No Passing											|
| Roadwork	      		| Roadwork						 				|
| Stop			| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


