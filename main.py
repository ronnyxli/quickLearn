# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:46:38 2018

@author: rli
"""

import os

from tkinter import filedialog

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats

from sklearn import datasets

# feature selection methods
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# model evaluation methods
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pdb



def normalize(df_in):
    '''
    z-score normalization - subtract mean, divide by std
    '''
    # replace each column with array of normalized values
    # TODO: check that each column is a pandas Series of numbers
    df_out = df_in.apply(lambda x: (x-np.mean(x))/np.std(x), axis=0)
    return df_out


def feature_selection(df_in):
    '''
    Determine features contributing to the greatest variance in the data
    '''
    selected_features = []

    feature_df = df_in.drop(['class'], axis=1)
    feature_df_norm = normalize(feature_df)

    unique_classes = df_in['class'].unique()

    # loop all features and generate array of class separation coefficients
    sep_coeff = []

    for curr_feature in feature_df.columns:
        # get array containing all observations of current feature
        feature_array = feature_df[curr_feature]
        # initialize dict to store the distribution function of the current feature for each class
        class_dist = []
        for curr_class in unique_classes:
            class_idx = df_in[df_in['class'] == curr_class].index
            # quantify distribution by mean and std (assume Gaussian)
            mu = np.mean(feature_array[class_idx])
            stdev = np.std(feature_array[class_idx])
            pdb.set_trace()
            class_dist.append({'mu':mu, 'std':stdev})
        # TODO: given all class distributions for current feature, calculate separation score
        pdb.set_trace()
        # percent overlap?
        sep_score = 0
        sep_coeff.append(sep_score)

    # calculate entropy for each feature
    entropy_df = feature_df.apply(lambda x: stats.entropy(x), axis=0)

    # TODO: remove correlated features

    return selected_features


def split_data(df_in):
    '''
    Split test data in training and validation sets
    '''

    num_features = df_in.shape[1]-1

    feature_arrays = df_in.values[:,0:num_features]
    label_array = df_in.values[:,-1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(feature_arrays,label_array,
                                                                                    test_size=validation_size,
                                                                                    random_state=seed)

    # place split data into dict
    data_out = {'training_data':X_train, 'training_labels':Y_train, 'validation_data':X_validation, 'validation_labels': Y_validation}

    return data_out


def model_selection(X_train, X_label):
    '''
    '''

    knn = KNeighborsClassifier()
    knn.fit(data_split['training_data'], data_split['training_labels'])
    predictions = knn.predict(data_split['validation_data'])
    print(accuracy_score(data_split['validation_labels'], predictions))
    print(confusion_matrix(data_split['validation_labels'], predictions))
    print(classification_report(data_split['validation_labels'], predictions))

    return 0



if __name__ == "__main__":

    # prompt user to select feature table
    # data_dir = filedialog.askopenfilename(title = "Select data file")
    # data = pd.read_csv(file_path)

    # construct dataframe from UCI ML Wine Data Set
    data = datasets.load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    for n in range(0,len(data.target_names)):
        data.target = [data.target_names[n] if x==n else x for x in data.target]
    df['class'] = pd.Series(data.target)

    # TODO: check whether dataframe is in supported format

    # rank and select the most relevant features
    df_filtered = feature_selection(df)

    # split data into training and validation sets
    df_split = split_data(df_filtered)

    # TODO: loop through models and find the best one

    # TODO: select best-performing model and output classifier
