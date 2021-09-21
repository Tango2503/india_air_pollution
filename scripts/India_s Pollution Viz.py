import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
#from sklearn.cross_validation import train_test_split

dataset = pd.read_csv('dataset.csv', low_memory = False)
def convert_date(x):
    try:
        return datetime.strptime(x, '%m/%d/%Y')
    except:
        return 'Exception Raised'

dataset['date_processed'] = dataset['date'].apply(convert_date)
dataset.loc[dataset['date_processed'] == 'Exception Raised']
# The dates here are either weird or missing. E.g. 31st September and 31st June. 
cleaned_data = dataset.loc[dataset['date_processed'] != 'Exception Raised']
cleaned_data['Year'] = cleaned_data['date_processed'].dt.year
cleaned_data['date_processed'] = cleaned_data['date_processed'].dt.date

def clean_type(t):
    if (t == 'Residential, Rural and other Areas') | (t == 'Residential and others') | (t == 'RIRUO') | (t == 'Residential'):
        return 'Residential'
    elif (t == 'Industrial Area') | (t == 'Industrial Areas') | (t == 'Industrial'):
        return 'Industrial'
    else:
        return 'Sensitive'

cleaned_data['type']  = cleaned_data['type'].apply(clean_type)


# =============================================================================
# """ If we consider India's yearly average pollutant readings per day and compare it with WHO’s quality 
# standards we can clearly see that the country is a breeding ground for airborne diseases (lovely). """
# 
# cleaned_data = cleaned_data.loc[cleaned_data['Year'] > 2010]
# daily_pollution_all = cleaned_data.groupby(['location','date_processed','Year'])['no2','so2','rspm','spm'].mean().reset_index()
# daily_pollution_all = daily_pollution_all.groupby(['date_processed','Year'])['no2','so2','rspm','spm'].mean().reset_index()
# yearly_pollution_average = daily_pollution_all.groupby(['Year'])['no2','so2','rspm','spm'].mean().reset_index()
# 
# =============================================================================

""" To understand the gravity of the situation, I want to see how worse off the non-attainment cities are to the other cities given in the dataset. """


non_attainment_cities = pd.read_csv('Non Attainment Cities.csv')

def isNonAttainment(city, non_attainment_cities):
    if city in non_attainment_cities['Cities'].unique().tolist():
        return 'Non Attainment'
    else:
        return 'Other'
    
cleaned_data['isNonAttainment'] = cleaned_data['location'].apply(lambda x: isNonAttainment(x, non_attainment_cities))
daily_pollution_all = cleaned_data.groupby(['isNonAttainment','date_processed','Year'])['no2','so2','rspm','spm'].mean().reset_index()
#daily_pollution_all = daily_pollution_all.groupby(['date_processed','Year'])['no2','so2','rspm','spm'].mean().reset_index()
yearly_pollution_average = daily_pollution_all.groupby(['Year','isNonAttainment'])['no2','so2','rspm','spm'].mean().reset_index()

fig = plt.figure(figsize=(9,9))
fig.suptitle('YOY Non Attainment vs Others', fontsize=14, fontweight='bold')
count = 0
for gas in ['no2','so2','rspm','spm']:
    count += 1
    ax = fig.add_subplot(2,2,count)
    fig.subplots_adjust(top=0.85)
    ax.set_title(gas)
    print(gas)
    one_gas = yearly_pollution_average[['Year','isNonAttainment',gas]]
    print(one_gas[gas].mean())
    one_gas = pd.pivot_table(one_gas, columns = 'isNonAttainment', values = gas, index = 'Year').reset_index()
    one_gas['Increase'] = (one_gas['Non Attainment'] - one_gas['Other'])/one_gas['Non Attainment']
    ax.text(one_gas['Year'].min(), one_gas['Other'].min(), '{:.2f} %'.format(100*one_gas['Increase'].mean()), style='italic')
    
    ax.plot(one_gas['Year'], one_gas[['Non Attainment','Other']], lw = 2)
    ax.legend(loc='upper left', frameon=False)

plt.show()   


""" Regression Analysis on RSPM and SPM """
# So for a particular month and location, you can have a type and a location monitoring station
# On a particular day, in a particular location and type of location, there can be multiple monitoring stations 
# Instead of summing the pollution, we take a mean across all stations to find the average air quality for that day.

# average_non_attainment = non_attainment_data.groupby(['location','date_processed','type'])['no2','total_pollution'].mean().reset_index()

# Gotta plot that on maps

non_attainment_data = cleaned_data.loc[cleaned_data['isNonAttainment']== 'Non Attainment']
daily_pollution = non_attainment_data.groupby(['location','date_processed','Year','type'])['no2','so2','rspm','spm','pm2_5'].mean().reset_index()
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=360).mean()
    rolstd = pd.Series(timeseries).rolling(window=360).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return rolmean



spm = daily_pollution.groupby(['date_processed'])['rspm','spm','pm2_5'].mean().reset_index()

# Try dropping nas
spm['Total'] = spm['rspm'].fillna(0) + spm['spm'].fillna(0) + spm['pm2_5'].fillna(0)
spm = spm[['date_processed','Total']]
spm = spm.dropna()
#plt.figure(figsize = [15,15])
spm = spm.loc[spm['date_processed'] >= date(2003,1,1)]
spm.set_index('date_processed', inplace=True)
ts = spm['Total']

# MachineLearning Mastery tutorial AutoCorrelation
from pandas.plotting import lag_plot
lag_plot(ts)
plt.show()

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ts)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ts, lags=31)
plt.show()

#
#
## Monthly Data
##spm['Month'] = spm['date_processed'].apply(lambda x: x.strftime('%Y-%m'))
##spm = spm.groupby(['Month'])['rspm','spm','pm2_5'].sum().reset_index()
### Try dropping nas
##spm['Total'] = spm['rspm'].fillna(0) + spm['spm'].fillna(0) + spm['pm2_5'].fillna(0)
##spm = spm[['Month','Total']]
##spm = spm.dropna()
###plt.figure(figsize = [15,15])
##spm = spm.loc[spm['Month'] >= '2003-01']
##spm.set_index('Month', inplace=True)
##ts = spm['Total']
#
#rolmean = test_stationarity(ts)
#detrended = ts - rolmean
#rolmean = test_stationarity(detrended.dropna())

expweighted_avg = ts.ewm(span=180, adjust=False).mean()
detrended = ts - expweighted_avg
rolmean = test_stationarity(detrended.dropna())

# Automatic STL decomposition 
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts, model = 'additive', freq = 1)
decomposition.plot()
plt.show()

import statsmodels.api as sm
#
#import seaborn as sns

# Set figure width to 12 and height to 9
fig = plt.figure(figsize = (12,9))
cycle, trend = sm.tsa.filters.hpfilter(ts, 30)
ax = fig.subplots(3,1)
ax[0].plot(ts)
ax[0].set_title('Data')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()

# Persistence Model 
component_dict = {'cycle': cycle, 'trend': trend}
prediction_results = []
for component in ['trend', 'cycle']:
    historic = component_dict[component].iloc[:int(len(ts) * 0.7)].tolist() # 70% used for training
    test = component_dict[component].iloc[int(len(ts) * 0.7):]
    predictions_persistence = []
    for i in range(len(test)):
        yhat = historic[-1]
        predictions_persistence.append(yhat)
    mse = mean_squared_error(test, predictions_persistence)
    rmse = np.sqrt(mse)
    print(component + ' RMSE of persistence model: %.3f' % rmse)

# Fit models
from statsmodels.tsa.ar_model import AR

component_dict = {'cycle': cycle, 'trend': trend}
prediction_results = []
for component in ['trend', 'cycle']:
    historic = component_dict[component].iloc[:int(len(ts) * 0.7)].tolist() # 70% used for training
    test = component_dict[component].iloc[int(len(ts) * 0.7):]
    predictions = []
    for i in range(len(test)):
        model = AR(historic)
        model_fit = model.fit()
        l = len(historic)
#        print(l)
        pred = model_fit.predict(start=l, end=l, dynamic=False)
        params = model_fit.params
        predictions.append(pred[0])
        historic.append(test[i])
    predictions = pd.Series(predictions, index=test.index, name=component)
    prediction_results.append(predictions)
    test_score = np.sqrt(mean_squared_error(test, predictions))
    print(f'Test for {component} MSE: {test_score}')
    # plot results
    plt.plot(test.iloc[:], label='Observed '+component)
    plt.plot(predictions.iloc[:], color='red', label='Predicted '+component)
    plt.legend()
    plt.show()

recomposed_preds = pd.concat(prediction_results,axis=1).sum(axis=1)
recomposed_preds.name = 'recomposed_preds'
plt.plot(ts.iloc[int(len(ts) * 0.7):], label='Observed')
plt.plot(recomposed_preds, color='red', label='Predicted')
plt.legend()
plt.show()
test_score = np.sqrt(mean_squared_error(ts.iloc[int(len(ts) * 0.7):], recomposed_preds))
print(f'RMSE: {test_score}')

# ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
 
# fit model
model = ARIMA(ts, order=(3,0,7))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())



    

# Impute Data
""" The data imputation strategy has to change """
""" This is because the median value doesn't represent the whole population (e.g. for low pollution areas) """
""" Also doesn't consider time regression analysis in the picture """


""" Could create an imputing function which will run on a groupby of location, year, (season?), type """

daily_pollution = daily_pollution.loc[daily_pollution['Year'] > 2003]

imputer = Imputer(missing_values="NaN", strategy="median", axis = 0)
imputer = imputer.fit(daily_pollution[['no2','so2','rspm','spm','pm2_5']])
daily_pollution[['no2','so2','rspm','spm','pm2_5']] = imputer.transform(daily_pollution[['no2','so2','rspm','spm','pm2_5']])
daily_pollution['total_pollution'] = daily_pollution['no2'] + daily_pollution['so2'] + daily_pollution['rspm'] + daily_pollution['spm'] + daily_pollution['pm2_5']
daily_pollution = pd.pivot_table(daily_pollution, values = 'total_pollution', index = ['location','date_processed','Year'], columns = ['type']).reset_index()

del daily_pollution['Sensitive']

# Impute Data
""" The data imputation strategy has to change """
""" This is because the median value doesn't represent the whole population (e.g. for low pollution areas) """
""" Also doesn't consider time regression analysis in the picture """

imputer = Imputer(missing_values="NaN", strategy="median", axis = 0)
# Dropping the Sensitive type of Area

#imputer = imputer.fit(daily_pollution[['Industrial', 'Residential', 'Sensitive']])
#daily_pollution[['Industrial', 'Residential', 'Sensitive']] = imputer.transform(daily_pollution[['Industrial', 'Residential', 'Sensitive']])
#daily_pollution['Total'] = daily_pollution['Industrial'] + daily_pollution['Residential'] + daily_pollution['Sensitive']
imputer = imputer.fit(daily_pollution[['Industrial', 'Residential']])
daily_pollution[['Industrial', 'Residential']] = imputer.transform(daily_pollution[['Industrial', 'Residential']])
daily_pollution['Total'] = daily_pollution['Industrial'] + daily_pollution['Residential']

# Seasonality

seasonality = daily_pollution.copy()
seasonality['Month'] = seasonality['date_processed'].apply(lambda x: x.strftime("%m"))
seasonality['Month'] = seasonality['Month'].apply(lambda x: int(x))
seasonality = seasonality.groupby(['Year','Month'])['Total'].sum().reset_index()
seasonality.plot(x = 'Month', y = 'Total', kind = 'scatter')

# Worst N cities of daily_pollution
worst30 = daily_pollution.groupby(['location'])['Total'].sum().reset_index().sort_values(by='Total', ascending = False).head(30)

# Getting their actual data (Daily)
worst30 = daily_pollution.loc[daily_pollution['location'].isin(worst30['location'].unique().tolist())]

# The plot shows that the contribution by Industries has reduced over the years and residential has shot up
cities = worst30['location'].unique().tolist()
cities_type = worst30.groupby(['Year','location'])['Industrial','Residential'].mean().reset_index()
plt.style.use('seaborn-white')
#plt.rcParams['figure.figsize'] = [20,20]
count = 0
plt.figure(figsize=(20,20))
for i in cities:
    count += 1
    plt.subplot(5,6,count)
    one_city = cities_type.loc[cities_type['location'] == i]
    industrial = list(one_city['Industrial'])
    residential = list(one_city['Residential'])
    year = list(one_city['Year'])
    plt.plot(year,industrial, label='Industrial', lw=2,)
    plt.plot(year,residential, label='Residential', lw=2)
    plt.title(i)
    plt.legend(loc='upper right')


def lr(predictor, daily_pollution):
    X = daily_pollution[[predictor]]#, 'Residential', 'Sensitive']]
    y = daily_pollution[['Total']]
    plt.figure(figsize=(5,5))
    
    model = LinearRegression().fit(X, y)
    print(predictor.upper())
    print("The coefficient is: " ,model.coef_)
    print("R^2 value is: " ,model.score(X,y)) 
    y_pred = model.predict(X)
    lin_mse = mean_squared_error(y_pred, y)
    lin_rmse = np.sqrt(lin_mse)
    print('Linear Regression RMSE: %.4f' % lin_rmse)
    plt.scatter(X,y)
    plt.plot(X, y_pred, color='red')
    plt.show()
    
lr('Industrial',worst30)
lr('Residential',worst30)
#lr('Sensitive',daily_pollution)

    
    
    


