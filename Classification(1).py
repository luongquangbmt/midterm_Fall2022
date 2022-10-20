#!/usr/bin/env python
# coding: utf-8

# In[20]:


## for data
import pandas as pd


import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for statistical tests
import scipy
#import statsmodels.formula.api as smf
#import statsmodels.api as sm

## for machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

# Import Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import Evaluation Metric
from sklearn.metrics import accuracy_score


## for explainer
#from lime import lime_tabular


#!pip3 install pandas


#https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec


from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay


# In[26]:


## for data
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LOAD DATA 
# Convert dataset to a pandas dataframe:
import pandas as pd

#John
#download ohiuchus directly from Vizier, ignore the last two columns (SIMBAD), ignore unit and blank row
dataset = pd.read_csv('ophiuchus_tablea1.tsv', delimiter=';',comment='#')
X = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED']]
y = dataset['Coretype']



#Tyler
#download ohiuchus directly from Vizier, ignore the last two columns (SIMBAD), ignore unit and blank row
dataset = pd.read_csv('lupus_tablea1.tsv', delimiter=';',comment='#')
X = dataset[['runNo', 'CoreName', 'RAJ2000', 'DEJ2000', 'Sig070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Sig160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Sig250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Sig350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Sig500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigNH2', 'NH2p', 'NH2p/NH2bg', 'NH2conv', 'NH2bg', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CuTExflag']]
y = dataset['Coretype']



#Nicholas
dataset = pd.read_csv('aquila_table1.tsv', delimiter=';',comment='#')
X = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag']]
y = dataset['Coretype']


# Sophia
dataset = pd.read_csv('taurus_table1.tsv', delimiter=';',comment='#')
X = dataset[['sig70', 'I70peak', 'e_I70peak',
       'C70', 'I70conv', 'S70', 'e_S70', 'a70', 'b70', 'PA70', 'sig160',
       'I160peak', 'e_I160peak', 'C160', 'I160conv', 'S160', 'e_S160', 'a160',
       'b160', 'PA160', 'sig250', 'I250peak', 'e_I250peak', 'C250', 'I250conv',
       'S250', 'e_S250', 'a250', 'b250', 'PA250', 'sig350', 'I350peak',
       'e_I350peak', 'C350', 'I350conv', 'S350', 'e_S350', 'a350', 'b350',
       'PA350', 'sig500', 'I500peak', 'e_I500peak', 'C500', 'S500', 'e_S500',
       'a500', 'b500', 'PA500', 'sigNH2', 'NH2peak', 'CNH2', 'NH2conv',
       'NH2bg', 'aNH2', 'bNH2', 'PANH2', 'Nsed']]
Y = dataset['CType']


#Sophie
dataset = pd.read_csv('cepheus_tablea1.tsv', delimiter=';',comment='#')
X = dataset[['Signi70', 'Sp70',
       'e_Sp70', 'Sp70/Sbg70', 'Sconv70', 'Stot70', 'e_Stot70', 'FWHMa70',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/NbgH2', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED', 'CSARflag']]
    
Y = dataset['Coretype']



#Troy
dataset = pd.read_csv('corona_australia_table1.tsv', delimiter=';',comment='#')
X1 = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070', 'FWHMb070',
       'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp/Sbg160', 'Sconv160',
       'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160', 'Signi250',
       'Sp250', 'e_Sp250', 'Sp/Sbg250', 'Sconv250', 'Stot250', 'e_Stot250',
       'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350', 'e_Sp350',
       'Sp/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350', 'FWHMa350', 'FWHMb350',
       'PA350', 'Signi500', 'Sp500', 'e_Sp500', 'Sp/Sbg500', 'Stot500',
       'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500', 'SigniNH2', 'NpH2',
       'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2', 'FWHMbNH2', 'PANH2', 'NSED',
       'CSARflag', 'CUTEXflag']]
    
Y = dataset['Coretype']
    
    
#Joey
dataset = pd.read_csv('orionb_table1.tsv', delimiter=';',comment='#')
X2 = dataset[['Signi070', 'Sp070', 'e_Sp070',
       'Sp070/Sbg070', 'Sconv070', 'Stot070', 'e_Stot070', 'FWHMa070',
       'FWHMb070', 'PA070', 'Signi160', 'Sp160', 'e_Sp160', 'Sp160/Sbg160',
       'Sconv160', 'Stot160', 'e_Stot160', 'FWHMa160', 'FWHMb160', 'PA160',
       'Signi250', 'Sp250', 'e_Sp250', 'Sp250/Sbg250', 'Sconv250', 'Stot250',
       'e_Stot250', 'FWHMa250', 'FWHMb250', 'PA250', 'Signi350', 'Sp350',
       'e_Sp350', 'Sp350/Sbg350', 'Sconv350', 'Stot350', 'e_Stot350',
       'FWHMa350', 'FWHMb350', 'PA350', 'Signi500', 'Sp500', 'e_Sp500',
       'Sp500/Sbg500', 'Stot500', 'e_Stot500', 'FWHMa500', 'FWHMb500', 'PA500',
       'SigniNH2', 'NpH2', 'NpH2/Nbg', 'NconvH2', 'NbgH2', 'FWHMaNH2',
       'FWHMbNH2', 'PANH2', 'NSED']]
    
Y = dataset['Coretype']





# In[27]:


# TASK: LOAD DATA THAT WAS GIVEN TO YOU
# Convert dataset to a pandas dataframe
dataset = pd.read_csv('iris.csv') 
print(dataset.columns)

# TASK: INSPECT DATA
# Plot histogram,boxplots, heatmap
# Check variable stasttics
# Check NaN values, etc.
# Visualize correlation


# TASK: DATA ENGINEERING, FEATURE ENGINEERING
# Use head() function to return the first 5 rows: 
print(dataset.head()) 
# Assign values to the X and y variables:
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values 
X = dataset[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = dataset['variety']

print("Summary Statistics of the X dataframe \n", X.describe())

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 


print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Standardizing variable
#Instruction: This step is optional
# You can use non, one or all of these scalers to find better model, 
# i.e. models with better accuracy & precision.
# For other scaling or feature engineering methods, check this ref:
# https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn+preprocessing#module-sklearn.preprocessing

# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train)
print("\n Summary Statistics of the X dataframe after Standard Scaling\n", X_train_df.describe())

# MinmaxScaling features by removing mean and scaling to unit variance:
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train)
print("\n Summary Statistics of the X dataframe after MinMaxScaler Scaling\n", X_train_df.describe())



# SELECT MODEL
#classifiers = [
#    KNeighborsClassifier(n_neighbors=3),
#    DecisionTreeClassifier(max_depth=5),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0)),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    MLPClassifier(alpha=1, max_iter=1000),
#    AdaBoostClassifier(),
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis(),
#]

# TASK:
# Initialize classifier
# Choose 3 classifers
# Check the paramters of these classifiers using sklearn manuals
# Change the paramters manualy or by using for loop to choose the better model (model with better metrics)


# Use the KNN classifier to fit data:
classifier = KNeighborsClassifier(n_neighbors=3)  

# Gaussian Process Classifier
classifier = DecisionTreeClassifier(max_depth=5)  

# Support Vector Machine
classifier = SVC(kernel="linear", C=0.025)

# Support Vector Machine
classifier = SVC(gamma=2, C=1)  

# Gaussian Process Classifier
classifier = GaussianProcessClassifier(1.0 * RBF(1.0)) 

# RandomForestClassifier
classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

# MLPClassifier
classifier = MLPClassifier(alpha=1, max_iter=1000)

# AdaBoostClassifier
classifier = AdaBoostClassifier()

# GnB
classifier = GaussianNB()  

# QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()


#TRAIN MODEL
classifier.fit(X_train, y_train)  # Train the classifier



# TASK: PREDICT NEW VALUES
# Predict y data with classifier: 
y_predict = classifier.predict(X_test)

#TASK: EVALUATE RESULTS
# Print results: 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 

# Evaluate label (subsets) accuracy
print(accuracy_score(y_test, y_predict))

print(y_test, y_predict)


# In[ ]:


# Tip for modularization
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py



#https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec


# In[ ]:





# In[47]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




