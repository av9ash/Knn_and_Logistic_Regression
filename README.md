# Knn_and_Logistic_Regression
Machine Learning Project for hand writing recognition with KNN and Logistic Regression from scratch in python for MNIST database.

Logistic Regression model and k-Nearest Neighbor (kNN) algorithm for MNIST handwritten digits recognition. 
The MNIST dataset contains 70,000 images of handwritten digits, divided into 60,000 images as training set and 10,000 images as test set. 
Each image has 28 × 28 pixels which are considered as the features of the image. 
Only third party libraries allowed are numpy, scipy and matplotlib.

KNN:

Creates class object with training data and labels.

Create batches of testing data as knn tends to fill memmory

Compute distances and store in a list of test images (unlabeled) from training images (labeled). Do this for all batches.

Predict the class of training image by taking vote from nearest neighbors.

Sort the distances list on basis of distance b/w testing point and all the training data.

Pick the first K least values from distance list

Assign the label of most common value in firt K values.


