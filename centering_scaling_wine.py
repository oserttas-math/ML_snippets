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

# model metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Load the dataset
df = pd.read_csv('white_wine.csv')
df.head()

# Create binary class if quality 5 or more assign 1 otherwise 0
f = lambda x: 1 if x>=5 else 0
df['bi_quality'] = df['quality'].map(f)
df

# Set up the pipeline
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the Pipeline
pipeline = Pipeline(steps)

# Predictors and Target= quality variables
X = df.drop(labels=['quality','bi_quality'], axis=1).values
y = df['bi_quality'].values

# Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                                    random_state=42)

# Fit the pipeline
knn_scaled_df = pipeline.fit(X_train, y_train)

# Instantiate and fit a KNN classifier to unscaled data
knn_unscaled_df = KNeighborsClassifier().fit(X_train, y_train)

# Compare scores
print('Accuracy with Scaled data: {}'.format(knn_scaled_df.score(X_test, y_test)))
print('Accuracy with Unscaled data: {}'.format(knn_unscaled_df.score(X_test, y_test)))



'''Accuracy with Scaled data: 0.964625850340136
Accuracy with Unscaled data: 0.9666666666666667'''
