'''
---------------------------------------------------------------------------------
                                imports
---------------------------------------------------------------------------------
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.compose import make_column_selector
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#load data
dataset = pd.read_csv('cepheus_tablea1.tsv',delimiter=';',comment='#') 
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
print(dataset.columns)
print(dataset)


def new_func(dataset):
    print(dataset.head()) 

#convert variables from string to float
column_selector=make_column_selector(dtype_include = object)
for col in column_selector(X):
    X[col].str.strip()
    X = X.replace(r'^\s*$', np.nan, regex=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
print(dataset.shape)
'''
-------------------------------------------------------------------------
                    Plot the data
-------------------------------------------------------------------------
'''
#plot histogram

dataset.plot(kind='hist', bins=80, density=True)
plt.show()

#plot boxplot

dataset.plot(kind='box',notch=True, patch_artist=True,showfliers=False)
plt.show()

#plot heatmap for correlation
corrmat = dataset.corr()
f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

'''
----------------------------------------------------------------------------
                   Find Statistics of data
----------------------------------------------------------------------------
'''

#check NaN values
def isNaN(num):
    return num!= num

data = float("nan")
print(isNaN(data))


#get statistics
d=dataset.describe(include='all')
print(d)
print(X.shape, Y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#get min max

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train)

#standardize

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train)
print("\n Summary Statistics of the X dataframe after Standard Scaling\n", X_train_df.describe())

'''
----------------------------------------------------------------------------
                    Choose classifiers and classify
----------------------------------------------------------------------------
'''

#classifiers
classifiers = [    
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    
    
]

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(dataset, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
'''
-------------------------------------------------------------------------------
                               Predict and Evaluate
-------------------------------------------------------------------------------
'''
#predict values
y_predict = classifiers[0].predict(X_test)

#TASK: EVALUATE RESULTS
# Print results: 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 

# Evaluate label (subsets) accuracy
print(accuracy_score(y_test, y_predict))

print(y_test, y_predict)
