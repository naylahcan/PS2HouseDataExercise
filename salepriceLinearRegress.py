import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('train.csv')
data.shape

#get sample of data for training
train = data.iloc[:,:]

#finding correlation factors with Sale Price
train['SalePrice']
numeric = train.select_dtypes(include=[np.number])
corr = numeric.corr()
cols = corr['SalePrice'].sort_values(ascending=False)[0:5].index

#initial predictions using train
X = train[cols]
Y = train['SalePrice']
X = X.drop(['SalePrice'], axis = 1)

#linear regression model
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X, Y)
predictions = model.predict(X)

predictFinal = []

for i in range(len(predictions)):
    predictFinal.append([i+1, predictions[i-1]])

np.savetxt('predictFile.csv', predictFinal, delimiter=',')