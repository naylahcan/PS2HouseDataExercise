import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#get sample of data for training
train = traindata.iloc[:,:]
test = testdata.iloc[:,:]

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
#data to test
cols = cols.drop(['SalePrice'])
XTest = test[cols]
predictions = model.predict(XTest)


#Test values
predictFinal = []
for i in range(len(predictions)):
    predictFinal.append([i+1001, predictions[i]])
print(predictFinal)
np.savetxt('predictFile2.csv', predictFinal, delimiter=',')

from sklearn.metrics import mean_squared_error

# mse = sklearn.metrics.mean_squared_error(train, predictFinal)

# rmse = math.sqrt(mse)