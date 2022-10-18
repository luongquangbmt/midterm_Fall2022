# midterm_Fall2022
Analizing the data:
The following data table cepheus.tsv was loaded into python to use different learning algorithms to classify the dataset.
When analyzing the statistics, it was necessary to scale the data since most of the statistics were Nan. After scaling all the means of each column ranges in between about -7 and 7. We can tell there was a large difference between the means. The standard deviation of the columns in the X data frame all resembled each other immensely, with a standard deviation of around 1.0007. The minimums of each column all had a relatively wide range of only negative numbers, while the maximum had a wide range of positive numbers. 
-	To continue the analysis of the columns, here is the histogram produced which we can can then see which columns have the highest frequency:
-	
 ![image](https://user-images.githubusercontent.com/113098596/196523616-f640891e-86a7-4f8d-b9c0-00ec2b7d4a1d.png)

-	Here is the box plot for analysis. The variables are difficult to visualize, but it goes in the order of the index: 
-	
![image](https://user-images.githubusercontent.com/113098596/196523673-17350b4d-8c3d-4b8c-a983-17ccd2ff7617.png)
 
-	And here is the heatmap of the correlation matrix, which Seq and CSARflag have the highest correlation of 1: 
-	
![image](https://user-images.githubusercontent.com/113098596/196523739-9997f51a-1a9f-48f0-8ca5-c4fe0694d6ac.png)

Classifiers:
After having analyzed the data by finding the statistics and plotting the data for more of a visual explanation, I then went and chose the models, with the best accuracy, used to classify the data:
-	Gaussian Naive Bayes: This model is especially useful when the whole dataset is too big to fit in memory at once. Naïve Bayes is a group of supervised machine learning classification algorithms based on the Bayes theorem. It is a simple classification technique, but has high functionality. The GaussianNB formula is:
![image](https://user-images.githubusercontent.com/113098596/196523845-125a65a6-2cfd-48f1-9ed2-c8b62f899753.png)
              
 The GaussianNB classifier gave an accuracy of 93%.
-	DecisionTreeClassifier: Decision trees work by splitting data into a series of binary decisions. These decisions allow you to traverse down the tree based on these decisions. You continue moving through the decisions until you end at a leaf node, which will return the predicted classification. The algorithm uses a number of different ways to split the dataset into a series of decisions. One of these ways is the method of measuring Gini Impurity.Here is the formula: 
    ![image](https://user-images.githubusercontent.com/113098596/196523885-9a2d8aaf-b7ff-45b0-ac52-ac5268734e39.png)
            
 The DecisionTreeClassifier gave an accuracy of 93%.
-	AdaBoostClassifier: AdaBoost classifier builds a strong classifier by combining multiple poorly performing classifiers so that you will get high accuracy strong classifier. The basic concept behind Adaboost is to set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations. 
-	The AdaBoostClassifier gave an accuracy of 93%.
-	Here below is the visual representations of the classifiers chose, showing that their accuracy is of 93%. 
-
![image](https://user-images.githubusercontent.com/113098596/196524028-bd179512-497d-46a4-86bf-cd09580b0112.png)


Splitting the data:
-	There was a process of splitting the data where the testing subset was changed to 30% and the training subset was assigned 70%. The following mathematical equations for the testing and training subsets are:
-	Testing:
-	P [|Ein − Eout| > ǫ ] ≤ 2 e−2ǫ 2N
-	Training:
-	P [|Ein − Eout| > ǫ ] ≤ 2M e−2ǫ 2N







Finding the accuracy:

 ![image](https://user-images.githubusercontent.com/113098596/196524097-cc648df0-e8fb-4b52-912e-0c38e60f8e5d.png)

Precision is defined as the proportion of cases found that were actually relevant.
Recall is defined as the proportion of the relevant cases that were actually found among all the relevant cases. You can see that recall is the same as sensitivity, because recall is also given by TP/(TP+FN).
Accuracy is defined as the ability of the classifier to select all cases that need to be selected and reject all cases that need to be rejected. For a classifier with 100% accuracy, this would imply that FN = FP = 0. Accuracy is given by (TP+TN)/(TP+FP+TN+FN).
 
The F1 score is the combination of precision and recall. The F1 score, to use better accuracy metrics, is another way to solve class imbalance problems, which consider not only the number of prediction errors that your model makes, but that also look at the type of errors that are made.
For the classification report:
                precision    recall  f1-score   support

           0       0.88      1.00      0.93       14
           1       1.00      0.88      0.93       16

   micro avg                           0.93       30
   macro avg       0.94      0.94      0.93       30
weighted avg       0.94      0.93      0.93       30


The accuracy found was 0.93333333, which means it was 93.333%, therefore highly accurate.
