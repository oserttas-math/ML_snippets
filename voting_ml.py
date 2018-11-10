# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Pipeline builder
from sklearn.pipeline import Pipeline

# Call Transfomers on duty
from sklearn.preprocessing import Imputer

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

# Cross validation from model selection
from sklearn.model_selection import cross_val_score

# model metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# import data
df = pd.read_csv('votes.csv', header = None)

# Preliminary Analysis
df.head(5)
df.info()
df.isnull().sum()

# Data have missing values but they are not nones
df[df.isin(['?'])==True]
df.isin(['?']).sum()

# Replace everything for machine learning algorithm
maps = {'?': np.nan, 'y':1, 'n':0}
new_df = df.replace(maps)
new_df

# Use Imputer to fill out nan values
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# Instantiate the SVC classifier
clf = SVC()

# Set up the pipeline
steps = [('imputation', imp), ('SVM',clf)]

# Create the Pipeline
pipeline = Pipeline(steps)

# Create training and test data
X = new_df.drop(labels=[0], axis=1).values
y = new_df[0].values

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                        test_size=0.3, random_state=42)

# Fit the pipeline to the train Set
pipeline.fit(X_train, y_train)

# Predict the labels
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

 precision    recall  f1-score   support

   democrat       0.99      0.96      0.98        85
 republican       0.94      0.98      0.96        46

avg / total       0.97      0.97      0.97       131
