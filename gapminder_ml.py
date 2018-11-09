# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pipeline builder
from sklearn.pipeline import Pipeline

# Call Transfomers on duty
from sklearn.preprocessing import Imputer

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

# Cross validation from model selection
from sklearn.model_selection import cross_val_score

# model metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# import data
df = pd.read_csv('gapminder.csv')

# Preliminary Analysis
df.head(5)
df.info()

# Create a boxplot of life expecrancy per region
fig, ax = plt.subplots(figsize=(8,10))
df.boxplot(column=['life'], by =['Region'], rot=60, ax=ax)
plt.show()
# Comment on Graph: What is going on Asia ?

# Create Dummy variables df_region
df_region = pd.get_dummies(df, columns=['Region'], drop_first=True)
df_region

# Create predictors X and target y values
X = df_region.drop(labels='life', axis=1).values
y = df_region['life'].values

# Regression with categorical variables
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross validation
ridge_cross = cross_val_score(ridge, X, y, cv=5)
print(ridge_cross)
