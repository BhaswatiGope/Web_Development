# Simple Linear Regression model using sample data to run through the prediction model on API using Python Flask

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sample_dataset = pd.DataFrame({"Experience":[9,6,5,2,7,3,10,11],
 "Score":[8,8,6,10,9,6,7,8],
 "Annual_Income":[50000, 45000, 60000, 65000,70000,62000,72000,80000]
 })

sample_dataset['Experience'].fillna(0, inplace=True)

sample_dataset['Score'].fillna(sample_dataset['Score'].mean(), inplace=True)

X = sample_dataset.iloc[:, :2]
y = sample_dataset.iloc[:, -1]


from sklearn.linear_model import LinearRegression
Model = LinearRegression()

#Fitting model with Training data

Model.fit(X, y)

# Saving model to disk
pickle.dump(Model, open('model.pkl','wb'))

# Loading model to compare the results

model_predict= pickle.load(open('model.pkl','rb'))
