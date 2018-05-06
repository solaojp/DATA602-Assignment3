#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:25:48 2018

@author: GD
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing ,cross_validation,svm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller


url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20180505"
uClient = uReq(url)
page_html = uClient.read()
page_soup = soup(page_html,"html.parser")
page_soup.find_all('tr',limit = 100)[0].find_all('th')
page_soup.find_all('th',{'class':"text-left"})[0].text
column_headers = [th.getText() for th in 
                  page_soup.findAll('tr', limit=100)[0].findAll('th')]
data_rows = page_soup.find_all("tr")[1:]
historic_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]
historic_data_02 = []
for i in range(len(data_rows)):
    historic_row = []
for td in data_rows[i].findAll('td'):
    historic_row.append(td.getText())
historic_data_02.append(historic_row)
historic_data == historic_data_02        
df = pd.DataFrame(historic_data, columns=column_headers)
#df.to_csv("Bitcoin.csv")
df['Date'] = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
df['Close'] = df.Close.astype(float)
df = pd.read_csv('Bitcoin.csv',index_col = 'Date',parse_dates = True)
ts = df['Close']
ts['2018-04-30':'2013-05-05']
plt.plot(ts)


#Function for testing stationarity

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=20)
    rolstd = pd.rolling_std(timeseries, window=20)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

### the Dickey-Fuller test is used to determine whether a unit root, a feature that can cause issues 
### in statistical inference, is present in an autoregressive model. 
### The formula is appropriate for trending time series like stock prices
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test static','P-Value','Lags', 'Number of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)


# making data stationary
ts_log = np.log(ts)
plt.plot(ts_log)


# Eliminating Trend
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log, freq=52)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


### smoothing

moving_avg = pd.rolling_mean(ts_log, 10, min_periods=1)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


### Exponentially weighted moving average

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


#Eliminating trend and seasonality

#Differencing:

#Take first difference:
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

#Final forecasting
### ACF & PACF Plots
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#### AR Model:

#AR model:
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


##MA model
model1 = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model1.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


### ARIMA model

model2 = ARIMA(ts_log, order=(5, 1, 0))  
results_ARIMA = model2.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS (Root Squared Sum): %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))



## Convert to original scale

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


plt.plot(ts_log)
plt.plot(predictions_ARIMA_log)


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA,start='2013-04-28', end='2018-05-25')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


