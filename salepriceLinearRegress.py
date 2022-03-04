import pandas as pandas
import numpy as pandas
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('train.csv')

#get sample of data for training
train = data.iloc(0:20,:)
train.head()
