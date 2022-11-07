

# Programming Challenge
# DD2421 VT22
# Henrik Fagerlund
# command for ubuntu: conda activate conda-scikit


# Import packages

import pandas as pd
import numpy as np
from numpy import savetxt
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing



# --------- Training ----------

# Read the training file
dataset = pd.read_csv('TrainOnMe-4.csv', index_col=0)


# Take out the y column and remove it from the training-set
y_train = dataset.y
dataset.drop(['y'], axis=1, inplace=True)

ley = preprocessing.LabelEncoder()
y_train = ley.fit_transform(y_train)


# Fit and transform all columns
le = preprocessing.LabelEncoder()
train = dataset.apply(le.fit_transform)


X_train = train


# # Train and fit a model to the training-dataset using auto-sklearn
# cls = classifier.AutoSklearnClassifier()
# cls.fit(X_train, y_train)

DT= DecisionTreeClassifier()
DT.fit(X_train,y_train)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)



# --------- Evaluation ----------

# Read the evaluation file
X_test = pd.read_csv('EvaluateOnMe-4.csv', index_col=0)

X_test = X_test.apply(le.fit_transform)

pred = DT.predict(X_test)
 
# # Predict y-values for the evaluation-dataset
# pred = cls.predict(X_test)

# Perform a inversed transform to get the y-values on the original form
y2 = ley.inverse_transform(pred)


# Save the predicted values to txt-file
np.savetxt('testPred.txt', y2, fmt='%s')

