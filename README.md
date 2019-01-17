# quickLearn
Trains and validates a basic machine learning model for classification given a table of feature vectors and corresponding labels.

Required python modules: numpy, pandas, matplotlib, sklearn

## Requirements for input data:
1) .csv format
2) Each row is an observation.
3) Contains a column titled "label" which contains the class labels.
4) All columns other than label" contain feature vectors.
5) All feature values are numeric.

## Execution
To run on a scikit-learn dataset (UCLA Wine Data Set), use the "python main.py" command. To run on a custom dataset contained in a csv file, use the "python main.py -m 1" command. You will be prompted to select the data file in an explorer window.

## Feature selection
For each feature vector, a one-way ANOVA is conducted to quantify the inter-class variance. The features are ranked by F-value and the features comprising 90% of the cumulative sum of F-values are retained.

## Model selection
The data is split into training and validation datasets.

The following models are evaluated using 10-fold cross-validation on the training set:
1) Logistic regression (LogR)
2) Linear discriminant analysis (LDA)
3) K-nearest neighbors (KNN)
4) Decision tree (CART)
5) Naive Bayes (NB)
6) Support vector machine (SVM)

The most accurate model is used to fit the training set and tested on the validation set.
