#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:42:36 2018

@author: ebrithil
"""

import pandas as pd
features = pd.read_csv('data/train_100k.csv')
labels = pd.read_csv('data/train_100k.truth.csv')
test_data = pd.read_csv('data/test_100k.csv')

features = features.values
labels = labels.values
test_data = test_data.values

X = features[:, 1:]
y = labels[:,1:]
test = test_data[:, 1:]
del(features)
del(labels)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#
# Neural Network 1
#
def slope_model():
    model = Sequential()
    model.add(Dense(32, input_dim=20, kernel_initializer='glorot_uniform', activation='selu'))
    model.add(Dense(64, input_dim=20, kernel_initializer='glorot_uniform', activation='selu'))
    model.add(Dense(128, input_dim=20, kernel_initializer='glorot_uniform', activation='selu'))
    model.add(Dense(256, input_dim=20, kernel_initializer='glorot_uniform', activation='selu'))
    model.add(Dense(512, input_dim=20, kernel_initializer='glorot_uniform', activation='selu'))
    
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model    

slope_estimator = KerasRegressor(build_fn=slope_model, epochs=300, batch_size=32, verbose=1)
slope_estimator.fit(X, y)
y_pred_training = slope_estimator.predict(X)
y_pred_test = slope_estimator.predict(test)

df_training = pd.DataFrame(y_pred_training)
df_test = pd.DataFrame(y_pred_test)

df_training.columns = ['slope', 'intercept']
df_test.columns = ['slope', 'intercept']

# Saving in a csv file the slope and intercept for both datasets
df_training.to_csv("env_keras/submission.train_100k.csv", index_label='id')
df_test.to_csv("env_keras/prediction.test_100k.csv", index_label='id')
