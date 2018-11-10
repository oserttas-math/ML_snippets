# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Pipeline builder
from sklearn.pipeline import Pipeline

# Call Transfomers on duty
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Cross validation from model selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# model metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



# Load the dataset
df = pd.read_csv('red_wine.csv', sep=';')
df.head()

# Create binary class if quality 5 or more assign 1 otherwise 0
f = lambda x: 1 if x>=5 else 0
df['bi_quality'] = df['quality'].map(f)
df

# Setup the pipeline
steps = [('scaler', StandardScaler()),('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
# C controls the regularization strength
# Gamma controls the kernel coefficient
parameters = {'SVM_C':[1, 10, 100],'SVM_gamma':[0.1, 0.01]}

# Create Target and Predictors
X = df.drop(labels=['quality', 'bi_quality'], axis=1).values
y = df['bi_quality'].values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=42)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
