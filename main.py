# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:46:38 2018

@author: rli
"""

import os
from tkinter import filedialog
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

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


def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser(description = "Run quickLearn")
    parser.add_argument("-m", metavar = "run mode", type = int, dest = "mode")
    return parser.parse_args()


def validate_data(df_in):
    '''
    Check whether dataframe is in supported format
    '''
    data_valid = False
    if len(df_in) > 0:
        if 'class' in df_in.columns:
            data_valid = True
    return data_valid


def normalize(df_in):
    '''
    z-score normalization - subtract mean, divide by std
    '''
    # replace each column with array of normalized values
    # TODO: check that each column is a pandas Series of numbers
    df_out = df_in.apply(lambda x: (x-np.mean(x))/np.std(x), axis=0)
    return df_out


def anova1(feature_vecs):
    '''
    Manual computation of one-way ANOVA
        Args: feature_vecs = N-length list of pandas Series where N = number of classes
        Out: F-value

        Assumptions:
            1. The samples are independent.
            2. Each sample is from a normally distributed population.
            3. The population standard deviations of the classes are all equal (i.e. homoscedastic).
    '''

    # calculate sum of squares for each class
    sum_x = []
    sum_x2 = []
    n = []
    SS = []
    for x in feature_vecs:
        sum_x.append(x.sum())
        sum_x2.append( (x**2).sum() )
        n.append(len(x))
        SS.append( (x**2).sum() - (x.sum()**2/len(x)) )

    # sum of squares
    SSb = np.sum(np.divide(sum_x2,n)) - np.sum(sum_x)**2/np.sum(n) # between
    SSw = np.sum(SS) # within
    SST = SSb + SSw # total

    # degrees of freedom
    DFb = len(feature_vecs) - 1 # between
    DFw = np.sum(n) - len(feature_vecs) # within
    DFT = DFb + DFw #total

    # mean squares
    MSb = SSb/DFb # between
    MSw = SSw/DFw # within

    # F-value
    F = MSb/MSw

    return F


def feature_selection(df_in):
    '''
    Rank and select the features contributing to the greatest variance in the data
        Args:
        Returns: Names of the selected features in a list
    '''

    selected_features = []

    feature_df = df_in.drop(['class'], axis=1)
    feature_df = normalize(feature_df)
    feature_names = feature_df.columns

    # TODO: find correlations between features

    # loop all features and generate array of class separation coefficients
    F_values = []

    for curr_feature in feature_names:
        # list of panda series representing class distributions for curr_feature
        samples = []
        for curr_class in df_in['class'].unique():
            # get array containing all observations of curr_feature for curr_class
            feature_array = feature_df[df_in['class'] == curr_class][curr_feature]
            samples.append(feature_array)
            '''
            sns.distplot(feature_array)

        sns.set_style('darkgrid')
        # plt.show()
        plt.savefig(curr_feature.replace('/','_') + '.png')
        plt.close()
        '''

        # 1-way ANOVA
        F = anova1(samples)
        # F,p = stats.f_oneway(samples)

        # TODO: calculate entropy for curr_feature?

        # save F-value for curr_feature
        F_values.append(F)

    # sort F-values in descending order (sort and flip)
    F_sorted = np.flip(np.sort(F_values), axis=0)
    F_sorted_idx = np.flip(np.argsort(F_values), axis=0)

    feature_names_sorted = feature_names[F_sorted_idx]

    # define cutoff index as 90% of the energy in the F-values
    cutoff_idx = np.argmax( np.cumsum(F_sorted)/F_sorted.sum() > 0.90 )

    # plot F-values vs. feature names
    selected = plt.bar( feature_names_sorted[0:cutoff_idx], F_sorted[0:cutoff_idx] )
    discarded = plt.bar( feature_names_sorted[cutoff_idx:], F_sorted[cutoff_idx:] )
    plt.xticks(rotation = 'vertical')
    plt.ylabel('F-value from one-way ANOVA')
    plt.legend( (selected, discarded),  ('Selected features', 'Discarded features') )
    plt.show()
    plt.close()

    return feature_names_sorted[0:cutoff_idx]


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

    df = []

    # get input arguments
    args_in = parse_args()

    if args_in.mode == 1:
        # prompt user to select feature table
        data_dir = filedialog.askopenfilename(title = "Select data file")
        df = pd.read_csv(file_path)
    else:
        # construct dataframe from UCI ML Wine Data Set
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        for n in range(0,len(data.target_names)):
            data.target = [data.target_names[n] if x==n else x for x in data.target]
        df['class'] = pd.Series(data.target)

    if validate_data(df):

        # select the most relevant features
        selected_features = feature_selection(df)

        X = df[selected_features] # features
        Y = df['class'] # class labels

        # split data into training and validation sets
        df_split = split_data(df_filtered)

        # TODO: loop through models and find the best one

        # TODO: select best-performing model and output classifier
