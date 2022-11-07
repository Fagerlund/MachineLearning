
# Programming Challenge
# DD2421 VT22
# Henrik Fagerlund
# command for ubuntu: conda activate conda-scikit


# Import packages
import pandas as pd
import numpy as np
from numpy import savetxt
import autosklearn.classification as classifier
from sklearn import preprocessing


# --------- Training ----------

# Read the training file
dataset = pd.read_csv('TrainOnMe-4.csv', index_col=0)

# Take out the y column and remove it from the training-set
y_train = dataset.y
dataset.drop(['y'], axis=1, inplace=True)

# Transform y
ley = preprocessing.LabelEncoder()
y_train = ley.fit_transform(y_train)

# Fit and transform all columns
le = preprocessing.LabelEncoder()
train = dataset.apply(le.fit_transform)

X_train = train

# Train and fit a model to the training-dataset using auto-sklearn
cls = classifier.AutoSklearnClassifier()
cls.fit(X_train, y_train)




# --------- Evaluation ----------

# Read the evaluation file
X_test = pd.read_csv('EvaluateOnMe-4.csv', index_col=0)
X_test = X_test.apply(le.fit_transform)
 
# Predict y-values for the evaluation-dataset
pred = cls.predict(X_test)

# Perform a inversed transform to get the y-values on the original form
y2 = ley.inverse_transform(pred)

# Save the predicted values to txt-file
np.savetxt('predicted.txt', y2, fmt='%s')




