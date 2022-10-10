import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for machine learning
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_selector as selector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import class_weight

# for metric evaluations
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


class MLModels:
    # Global attributes for convenience (encoder is global as it is used in several methods)
    scalerTypes = ['standard', 'minmax', 'robust']
    algorithms = ['GaussNB', 'Perceptron', 'SVM', 'NearestNeighbors']
    __encoder = preprocessing.LabelEncoder()

    # Constructor method opens, assigns, and gives a basic summary of data file given in parameter
    def __init__(self, pd_dataset, x_columns, y_columns):
        self.dataset = pd_dataset
        self.x = self.dataset[x_columns]
        self.y = self.dataset[y_columns]

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.x_train_df = None

        self.y_pred = None
        self.y_full_weights = None
        self.y_weights = None

    def summary_stats(self):
        print(f'Summary Stats of Loaded Dataframe \n {self.x.describe} \n {self.y.describe}')
        # print summary statistics of dataframe
        print(self.dataset.describe(include='all'))

    def dataset_plot(self):
        self.dataset.hist(figsize=(20, 20))

    def heat_map(self):
        # produce heatmap
        pear_corr = self.dataset.corr(method='pearson')
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pear_corr, interpolation='nearest')
        fig.colorbar(im, orientation='vertical', fraction=0.05)

    # Method preprocessors the data
    def clean_data(self):
        # stripping whitespace and replacing empty values with NaN
        cat_col_sel = selector(dtype_include=object)
        for col in cat_col_sel(self.x):
            self.x[col].str.strip()
        self.x = self.x.replace(r'^\s*$', np.nan, regex=True)

        # imputing missing data (x) values
        imputer = KNNImputer(missing_values=np.nan, add_indicator=True, n_neighbors=10)
        self.x = imputer.fit_transform(self.x)

        # encoding class (y) values
        self.y = self.__encoder.fit_transform(self.y)

    # method splits the data into a training set and testing set based on parameter
    def split_test_train(self, ratio):
        # split data and print shape of train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=ratio)
        print(f'Shape of original Dataframe: {self.x.shape} {self.y.shape} \n\
    Shape of training data: {self.x_train.shape} {self.y_train.shape} \n\
    Shape of testing data: {self.x_test.shape} {self.y_test.shape}')

        # calculating and creating list of class weights
        self.y_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train),
                                                           y=self.y_train)
        self.y_full_weights = []
        for value in self.y_train:
            self.y_full_weights.append(self.y_weights[value])

    # method scales the data based on the parameter given
    def scale_data(self, scale_type):
        # Rudimentary input validation
        while scale_type not in self.scalerTypes:
            scale_type = input(f'\nPlease select a valid scaler type: {self.scalerTypes}')

        # selecting scaler object to utilize
        if scale_type == 'standard':
            self.__scaler = preprocessing.StandardScaler()
        elif scale_type == 'minmax':
            self.__scaler = preprocessing.MinMaxScaler()
        elif scale_type == 'robust':
            self.__scaler = preprocessing.RobustScaler(with_centering=True, unit_variance=True)

        # scaling x data
        print(f'Scaling test and training x data using {scale_type}')
        self.__scaler.fit(self.x_train)
        self.x_train = self.__scaler.transform(self.x_train)
        self.x_test = self.__scaler.transform(self.x_test)
        self.x_train_df = pd.DataFrame(self.x_train)
        print(f'\nSummary of dataframe scaled with {scale_type}:')
        print(self.x_train_df.describe)

    # method chooses the algorithm to use for the model based on parameter given
    def classify_data(self, algorithm):
        if algorithm == 'NearestNeighbors':
            knn = KNeighborsClassifier(n_neighbors=6)
            knn.fit(self.x_train, self.y_train)
            self.y_pred = knn.predict(self.x_test)

        if algorithm == 'GaussNB':  # crappy most times
            gnb = GaussianNB(var_smoothing=1e-9)
            iso_calib = CalibratedClassifierCV(gnb, method='isotonic', ensemble=True)
            iso_calib.fit(self.x_train, self.y_train, sample_weight=self.y_full_weights)
            self.y_pred = iso_calib.predict(self.x_test)

        elif algorithm == 'Perceptron':
            args_tests = (
                {
                    "penalty": "l1",
                    "alpha": 1e-5,
                    "fit_intercept": False,
                    "shuffle": False,
                    "class_weight": "balanced"
                },
                {
                    "penalty": None,
                    "alpha": 1e-5,
                    "fit_intercept": False,
                    "shuffle": False,
                    "class_weight": "balanced"
                }, {
                    "penalty": None,
                    "alpha": 0.001,
                    "fit_intercept": False,
                    "shuffle": False,
                    "class_weight": "balanced"
                }, {
                    "penalty": "l1",
                    "alpha": 1e-7,
                    "fit_intercept": False,
                    "shuffle": False,
                    "class_weight": "balanced"
                }
            )
            pt = Perceptron(**args_tests[0])
            iso_calib = CalibratedClassifierCV(pt, method='isotonic', ensemble=False)
            iso_calib.fit(self.x_train, self.y_train)
            self.y_pred = iso_calib.predict(self.x_test)

        # SVM
        elif algorithm == 'SVM':
            args_tests = (
                {
                    "C": 1,
                    "kernel": "poly",
                    "degree": 3,
                    "gamma": "auto",
                    "shrinking": True,
                    "class_weight": "balanced"
                }, {
                    "C": 2,
                    "kernel": "poly",
                    "degree": 1,
                    "gamma": "auto",
                    "shrinking": True,
                    "class_weight": "balanced"
                }, {
                    "C": 1,
                    "kernel": "poly",
                    "degree": 3,
                    "gamma": "auto",
                    "shrinking": False,
                    "class_weight": "balanced"
                }, {
                    "C": 1,
                    "kernel": "poly",
                    "degree": 3,
                    "gamma": "auto",
                    "shrinking": True,
                    "class_weight": "balanced"
                }, {
                    "C": 3,
                    "kernel": "poly",
                    "degree": 2,
                    "gamma": "auto",
                    "shrinking": True,
                    "class_weight": "balanced"
                },

            )
            my_svm = svm.SVC(**args_tests[0])
            iso_calib = CalibratedClassifierCV(my_svm, method='isotonic', ensemble=False)
            iso_calib.fit(self.x_train, self.y_train)
            self.y_pred = iso_calib.predict(self.x_test)

        # method analyzes predicted values generated from classifyData. Prints Confusion matrix, classification
        # report, and overall accuracy of the algorithm
    def show_results(self):
        print(f'Confusion Matrix & classification report :'
              f'\n{confusion_matrix(self.y_test, self.y_pred)}'
              f'\n{classification_report(self.y_test, self.y_pred)}')

        # Evaluate label (subsets) accuracy
        acc_score = accuracy_score(self.y_test, self.y_pred)
        return acc_score
