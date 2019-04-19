# Knn_and_Logistic_Regression
Machine Learning Project for hand writing recognition with KNN and Logistic Regression from scratch in python for MNIST database.

Logistic Regression model and k-Nearest Neighbor (kNN) algorithm for MNIST handwritten digits recognition. 
The MNIST dataset contains 70,000 images of handwritten digits, divided into 60,000 images as training set and 10,000 images as test set. 
Each image has 28 × 28 pixels which are considered as the features of the image. 
Only third party libraries allowed are numpy, scipy and matplotlib.

KNN:

Steps:
Creates class object with training data and labels.

Create batches of testing data as knn likes to fill memory

Compute distances and store in a list of test images (unlabeled) from training images (labeled). Do this for all batches.

Predict the class of training image by taking vote from nearest neighbors.

Sort the distances list on basis of distance b/w testing point and all the training data.

Pick the first K least values from distance list

Assign the label of most common value in first K values.

Logistic Regression:

A binary logistic model has a dependent variable with two possible values, such as pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1". It can be used to classify non binary variables as well. To extend it to mulinomial variables we just pretend that the variable under consideration belongs to one class and the rest of the variables belong to the other class.

If the multinomial logit is used to model choices, it relies on the assumption of independence of irrelevant alternatives (IIA), which is not always desirable. This assumption states that the odds of preferring one class over another do not depend on the presence or absence of other "irrelevant" alternatives. For example, the relative probabilities of taking a car or bus to work do not change if a bicycle is added as an additional possibility. This allows the choice of K alternatives to be modeled as a set of K-1 independent binary choices, in which one alternative is chosen as a "pivot" and the other K-1 compared against it, one at a time.


Steps:
Create class object with alpha and number of iterations to be performed. Also initialize an empty weight metric.

Add an extra column with all values 1 in starting of the data matrix

For each label:

  Create a new list of labels such that each position where current label appears contains value 1 otherwise set it to 0

  Calculate the gradient ascent for this new label list and given data matrix
 
To calculate gradient ascent:

Create a new weights list will all items set to 1 and length equal to that of number of rows in data matrix

Calculate the dot product of the data matrix and this weights list

Calculate the sigmoid function of this output of the above dot product

Calcuate the error by subtracting the sigmoid from 1 present in the label copy list. Thus assigning the weights to each position of current label

Get the change in weight by multiplying alpha (learning rate) with dot product of errors and data matrix.

Update the weights list.
