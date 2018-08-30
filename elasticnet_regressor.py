"""
    Interview for Deep Learning Position
    
    Author: Jean Massucatto
    Computer Engineer 
"""

import pandas as pd
import numpy as np

#
#   LOAD DATASETS
#
# Comment and uncomment the line below to change from training data and test data
#features = pd.read_csv('data/train_100k.csv')
features = pd.read_csv('data/test_100k.csv')
labels = pd.read_csv('data/train_100k.truth.csv')

#Getting values for a matrix
features = features.values
features = features[:,1:]
labels = labels.values
labels = labels[:,1:]

## >>> Normalizing data
#from sklearn.preprocessing import StandardScaler
#stdsc = StandardScaler()
#features = stdsc.fit_transform(features)
#labels = stdsc.fit_transform(labels)

from sklearn.linear_model import ElasticNet
slope_inter = np.zeros([len(features),2])
for j in range(0,len(features)):

    X = []
    y = []
    for i in range (0,19,2):
        X.append(features[j,i])
        y.append(features[j,i+1])

    X = np.asarray(X)
    X = np.reshape(X,(len(X),1))
    y = np.asarray(y)


    # Creating and Fiting the linear model
    model = ElasticNet()
    model.fit(X,y)
    
    # Get the parameters
    slope_inter[j,0] = model.coef_
    slope_inter[j,1] = model.intercept_

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(labels[0], slope_inter[0])
mae = mean_absolute_error(labels[1], slope_inter[1])
print('Slope mse: ', mse) 
print('Intercept mae: ', mae)


df = pd.DataFrame(slope_inter)
df.head()
df.columns = ['slope', 'intercept']

# Saving in a csv file the slope and intercept for both datasets
# Just comment and uncomment the right line below
#df.to_csv("env/submission.train_100k.csv", index_label='id')
df.to_csv("env/prediction.test_100k.csv", index_label='id')
