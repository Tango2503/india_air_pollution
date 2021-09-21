#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:15:54 2019

@author: tanushagoswami
"""

import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import numpy as np


spm = pd.read_csv('SPM Pollution.csv')
spm['date_processed'] = spm['date_processed'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
## Monthly Data
#spm['Month'] = spm['date_processed'].apply(lambda x: x.strftime('%Y-%m'))
#spm = spm.groupby(['Month'])['rspm','spm','pm2_5'].sum().reset_index()
## Try dropping nas
#spm['Total'] = spm['rspm'].fillna(0) + spm['spm'].fillna(0) + spm['pm2_5'].fillna(0)
#spm = spm[['Month','Total']]
#spm = spm.dropna()
##plt.figure(figsize = [15,15])
#spm = spm.loc[spm['Month'] >= '2003-01']
#spm.set_index('Month', inplace=True)
#ts = spm['Total']


# Try dropping nas
spm['Total'] = spm['rspm'].fillna(0) + spm['spm'].fillna(0) + spm['pm2_5'].fillna(0)
spm = spm[['date_processed','Total']]
spm = spm.dropna()
#plt.figure(figsize = [15,15])
spm = spm.loc[spm['date_processed'] >= '2011-01-01']
spm.set_index('date_processed', inplace=True)
ts = spm['Total']

'''
# Quick check for Autocorrelation
lag_plot(ts, lag = 1)
plt.show()

plt.figure(figsize=(10,20))
plot_acf(ts, lags=30)
plt.show()
'''


# Persistence Model
def model_persistence(x):
	return x

# create lagged dataset
dataframe = pd.concat([spm.shift(1), spm], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-30], X[len(X)-30:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score) # 509.012
# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()


# AR Model
import statsmodels.api as sm
cycle, trend = sm.tsa.filters.hpfilter(ts, 30)
fig = plt.figure(figsize = (12,9))
ax = fig.subplots(3,1)
ax[0].plot(ts)
ax[0].set_title('Data')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()
from statsmodels.tsa.ar_model import AR

train, test = ts[1:int(len(ts) * 0.96)], ts[int(len(ts) * 0.96):]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

#component_dict = {'cycle': cycle, 'trend': trend}
#prediction_results = []
#for component in ['trend', 'cycle']:
#    historic = component_dict[component].iloc[:int(len(ts) * 0.7)].tolist() # 70% used for training
#    test = component_dict[component].iloc[int(len(ts) * 0.7):]
#    predictions = []
#    model = ARIMA(historic, order = (2,1,0))
#    model_fit = model.fit()
#    pred = model_fit.predict(start=len(historic), end=len(historic)+len(test)-1, dynamic=False)
#    params = model_fit.params
#    predictions = pd.Series(pred, index=test.index, name=component)
#    prediction_results.append(predictions)
#    test_score = np.sqrt(mean_squared_error(test, predictions))
#    print(f'Test for {component} MSE: {test_score}')
#    # plot results
#    plt.plot(test.iloc[:], label='Observed '+component)
#    plt.plot(predictions.iloc[:], color='red', label='Predicted '+component)
#    plt.legend()
#    plt.show()


