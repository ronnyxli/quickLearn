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

from scipy.io import loadmat

from sklearn import datasets

# feature selection methods
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



def normalize(df):
    '''
    z-score normalization - subtract mean, divide by std
    '''
    # replace each column with normalized values
    for column in df:
        if type(df[column][0]) is not str:
            df[column] = (df[column] - np.mean(df[column])) / np.std(df[column])
    
    return df


def feature_selection(df):
    '''
    Use PCA to determine features contributing to the great variance in the data
    '''
    selected_features = []

    df_norm = normalize(df)
    
    # separate features from labels
    num_features = df_norm.shape[1]-1
    feature_names = df_norm.columns[0:num_features]
    feature_arrays = df_norm.values[:,0:num_features]
    label_array = df_norm.values[:,-1]

    pca = PCA(n_components=0.95)
    fit = pca.fit(feature_arrays)
    cc = fit.components_
    
    # summarize components
    cc_array = np.apply_along_axis(np.abs, 0, cc).sum(axis=0)

    # TODO: remove correlated features
    

    return selected_features


def split_data(df):
    '''
    Split test data in training and validation sets
    '''
    
    num_features = df.shape[1]-1
    
    feature_arrays = df.values[:,0:num_features]
    label_array = df.values[:,-1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(feature_arrays,label_array,
                                                                                    test_size=validation_size,
                                                                                    random_state=seed)
    
    # place split data into dict
    data_out = {'training_data':X_train, 'training_labels':Y_train, 'validation_data':X_validation, 'validation_labels': Y_validation}
        
    return data_out




if __name__ == "__main__":
    
    # data_dir = filedialog.askopenfilename(title = "Select data file")
    file_path = 'acc_training_data.csv'
    
    data = pd.read_csv(file_path)
    
    data = datasets.load_iris()
    
    # TODO: check whether dataframe is valid
    
    # rank and select the most relevant features
    data_filtered = feature_selection(data)
    
    # split data into training and validation sets
    data_split = split_data(data)
    
    # TODO: loop through models and find the best one

    # TODO: select best-performing model and output classifier
    
    knn = KNeighborsClassifier()
    knn.fit(data_split['training_data'], data_split['training_labels'])
    predictions = knn.predict(data_split['validation_data'])
    print(accuracy_score(data_split['validation_labels'], predictions))
    print(confusion_matrix(data_split['validation_labels'], predictions))
    print(classification_report(data_split['validation_labels'], predictions))
    
    