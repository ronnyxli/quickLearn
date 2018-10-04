# quickLearn
Trains and validates an optimal machine learning model given a table of feature vectors and corresponding labels.

Required python modules: numpy, pandas, matplotlib, sklearn

## Requirements for input data:
1) .csv format
2) Each row is an observation.
3) Contains a column titled "class" which contains the class labels.
4) All columns other than "class" contain feature vectors.
5) All feature values are numeric.

## Feature selection
For each feature vector, a one-way ANOVA is conducted to quantify the inter-class variance. The features are ranked by F-value and the features comprising 90% of the cumulative sum of F-values are retained.

## Model selection
The data is split into training and validation datasets.

The following models are evaluated using 10-fold cross-validation on the training set:
1) Logistic regression (LR)
2) Linear discimant analysis (LDA)
3) K-nearest neighbors (KNN)
4) Decision tree (CART)
5) Naive Bayes (NB)
6) Support vector machine (SVM)

The most accurate model is used to fit the training set and test the validation set.
