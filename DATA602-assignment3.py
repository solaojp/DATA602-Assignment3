
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:25:27 2018

@author: GD
"""
### Importing libraries
 
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import warnings
warnings.filterwarnings("ignore")
from time import gmtime, strftime
from tabulate import tabulate
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing ,cross_validation,svm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

### List creation

hlist = []
pllist = []
cur_bal = []
old_Price = []
Quantity = []
Ticker = []
Old_wap = []


# Functions building blotter and P/L


# Base returns multiplication of the amount you want to trade  with current price   
def base(input1,input2):
    url="https://api.coinmarketcap.com/v1/ticker/"
    condata = requests.get(url).json()
    for x in condata:
        if x["name"]== input1:
            price  = x["price_usd"]
            price = int(float(price))
            total = (price) * int(input2)
            return(total)
           

# Executed price store the value at which trade was executed.I
            
def Executed_Price(input1):
    url="https://api.coinmarketcap.com/v1/ticker/"
    condata = requests.get(url).json()
    for x in condata:
        if x["name"]== input1:
            price  = x["price_usd"]
            price = (float(price))
            old_Price.append(price)
            return(price)

# Returns ticker or trade short form
            
def tick(input1):
    url="https://api.coinmarketcap.com/v1/ticker/"
    condata = requests.get(url).json()
    for x in condata:
        if x["name"]== input1:
            tick = x["symbol"]
            Ticker.append(tick)
            return(tick)
  

            
            
def Market():
    url="https://api.coinmarketcap.com/v1/ticker/"
    condata = requests.get(url).json()
    for x in condata:
        if x["name"]== input1:
            price  = x["price_usd"]
            Market_Price = (float(price))
            return(Market_Price)
            
 
### TS is time stamp                       
def TS(input1):
    TimeStamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return(TimeStamp)

# updatedprice returns ask price and bid price required for trading .n order for a transaction to occur, 
# someone must either sell to the buyer at the lower (Bid) price, or someone must buy from the sell 
# at the higher (Ask) price.
    
def updatedprice(input1):
    ticker="USDT-"+ tick(input1)
    url="https://bittrex.com/api/v1.1/public/getorderbook?market="+ticker+"&type=both"
    jdata=requests.get(url).json()
    data1=jdata["result"]["sell"][0]
    askprice=data1['Rate']
    data2=jdata["result"]["buy"][0]
    bidprice=data2['Rate']
    price=[askprice,bidprice]
    askprice = price[0]
    bidprice = price[1]
    return(askprice)

# Realised profit
    
def Rpl(input1,input2):
    askprice = updatedprice(input1)
    E_Price = Executed_Price(input1)
    Rpl = (askprice - E_Price) * (float(input2))
    return(Rpl)
        
# WAP
    
def Wap(input1):
    sum_q = sum(Quantity)
    sum_old_wap = sum(Old_wap)
    wap = ((sum_old_wap)) / (sum_q)
    return(wap)
 
# Current balance
    
def cur_bal(input1):
    total = base(input1,input2)
    cur_bal = 100000000.00
    done = False
    while not done:
        
       if (selected == '1'):
           done = False
           cur_bal = cur_bal - float(total)
           cur_bal = cur_bal
           break
       elif (selected == '2'):
            done = False
            cur_bal = cur_bal + float(total)
            cur_bal = cur_bal
            break
            
    return(cur_bal)
        


# Graph for rolling average and standard deviation

def normal_graph(input1):
    url = ("https://coinmarketcap.com/currencies/"+input1+"/historical-data/?start=20130428&end=20180505")
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
    df['Date'] = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    df['Close'] = df.Close.astype(float)
    df = pd.read_csv('Bitcoin.csv',index_col = 'Date',parse_dates = True)
    ts = df['Close']
    ts['2018-05-01':'2013-04-28']
    plt.plot(ts)
    plt.show(block = False)
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=20)
    rolstd = pd.rolling_std(ts, window=20)
    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    ### the Dickey-Fuller test is used to determine whether a unit root, a feature that can cause issues  
    ### The formula is appropriate for trending time series like stock prices
    #Perform Dickey-Fuller test:
    
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test static','P-Value','Lags', 'Number of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
        
    return(ts)
        
    
## Machine learning model I
    
def ARIMAmodel(input1):
    ts = normal_graph(input1)
    ts_log = np.log(ts)
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF:    
    plt.subplot(121)    
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.show(block = False)

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show(block = False)
    
    
def ARMODEL(input1):
    ts = normal_graph(input1)
    ts_log = np.log(ts)
    ts_log_diff = ts_log - ts_log.shift()
    model = ARIMA(ts_log, order=(2, 1, 0))  
    results_AR = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
    plt.show(block = False)
    return(results_AR)
    
    
    
def ARPredictions(input1): 
    url = ("https://coinmarketcap.com/currencies/"+input1+"/historical-data/?start=20130428&end=20180505")
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
    df['Date'] = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    df['Close'] = df.Close.astype(float)
    df = pd.read_csv('Bitcoin.csv',index_col = 'Date',parse_dates = True)
    ts = df['Close']
    ts['2013-04-28':'2018-05-05']
    
    ts_log = np.log(ts)
    model = ARIMA(ts_log, order=(2, 1, 0))  
    results_AR = model.fit(disp=-1) 
    
    predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
    #print (predictions_ARIMA_diff.head())
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    #print (predictions_ARIMA_diff_cumsum.head())
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    #predictions_ARIMA_log.head()
    #plt.plot(ts_log)
    #plt.plot(predictions_ARIMA_log)
    predictions_ARIMA = np.exp(predictions_ARIMA_log).tail(5)
    #predictions_ARIMA = predictions_ARIMA[::-1]
    #plt.plot(predictions_ARIMA,start='2013-04-28', end='2018-05-05')
    #plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    print(predictions_ARIMA)
    
    
# Machine learning model II
    
def LR(input1):
    url = ("https://coinmarketcap.com/currencies/"+input1+"/historical-data/?start=20130428&end=20180505")
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
    df = pd.DataFrame(historic_data,columns=column_headers)
    df['Date'] = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    df.set_index('Date')
    df['Close'] = df.Close.astype(float)
    df['Close'] = pd.to_numeric(df['Close'])
    df = df[['Close']]
    df = df.iloc[::-1]
    forecast_out = int(1)
    df['prediction'] = df[['Close']].shift(-forecast_out)
    #print(df.tail(10))
    # defining Features and Label
    x = np.array(df.drop(['prediction'],1))
    x = preprocessing.scale(x)
    x_forecast = x[-forecast_out:]
    x = x[:-forecast_out]
    y = np.array(df['prediction'])
    y = y[:-forecast_out]
    # Linear Regression
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
    # Training
    clf = LinearRegression()
    clf.fit(x_train,y_train)
    # Testing
    confidence = clf.score(x_test,y_test)
    print("confidence:",confidence)
    LR_prediction = clf.predict(x_forecast)
    # Residual Plots
    plt.scatter(clf.predict(x_train),clf.predict(x_train) - y_train,c = 'b',s = 40,alpha = 0.5)
    plt.scatter(clf.predict(x_test),clf.predict(x_test) - y_test,c = 'g',s = 40)
    plt.hlines(y = 0,xmin = 0,xmax = 50)
    plt.title('Residual Plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.show(block = False)
    return(LR_prediction)
    

    
def buy(input2):
    #cur_bal = 100000000.0
    ticker1 = tick(input1)
    Price = Executed_Price(input1)
    #total = base(input1,input2)
    #cur_bal = cur_bal(input1)
    order1 = {"Side":"Buy","Ticker":ticker1,"Quantity":input2,"Executed_price":Price,"Timestamp":TS(input1),"Current Balence":cur_bal(input1)}
    hlist.append(order1)
    pllist1 ={"Ticker":ticker1,"Quantity":input2,"Market":Market(),"Wap":Wap(input1),"Upl":base(input1,input2),"Rpl":0,"LR_prediction":LR(input1)} 
    pllist.append(pllist1)
    return(order1)
    
    
def sell(input2):
    #cur_bal = 100000000.0
    ticker2 = tick(input1)
    Price = Executed_Price(input1)
    #total = base(input1,input2)
    #cur_bal = cur_bal(input1)
    order2 = {"Side":"sell","Ticker":ticker2,"Quantity":input2,"Executed_price":Price,"Timestamp":TS(input1),"Current Balence":cur_bal(input1)}
    hlist.append(order2)
    pllist2 ={"Ticker":ticker2,"Quantity":input2,"Market":Market(),"Wap":Wap(input1),"Upl":0,"Rpl": Rpl(input1,input2),"LR_prediction":LR(input1)}  
    pllist.append(pllist2)
    return(order2)
    
            
    
def pieplot(Blotter):
    labels = Blotter['Ticker']
    sizes = Blotter['Executed_price']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show(block = False)
    

# Menu asks you to choose from following options
# press 1 to start the trade
    
def menu():
    print("     ")
    print("Hello!! How can we help you?Please select from the following options!")
    print("     ")
    print("1. Trade")
    print("2. Show Blotter")
    print("3. Show P/L" )
    print("4. Quit")
    
done = False
while not done:
    menu()
    selected = input()
    if selected == "1":
        done = False
        print("Welcome!Enter the name of equity you want to trade!. for Ex: If you want to trade 'Bitcoin' : Enter - Bitcoin ")
        def submenu():
            print("       ")
            print("1.Bitcoin")
            print("2.Ethereum")
            print("3.Ripple")
            print("4.Bitcoin Cash")
            print("5.Litecoin")
        done = False
        while not done:
            submenu()
            input1 =str(input())
            print("Enter quantity you want to trade")
            input2 = int(input())
            Quantity.append(input2)
            Old_wap.append(base(input1,input2))
            print("      ")
            #print("time series for executed_price history")
            #normal_graph(input1)
            #test_stationarity(ts)
            print("What is your trade option today?")
            print("     ")
            print("1.Buy")
            print("2.Sell")
            selected = input()
            if selected == "1":
                if input2 == 0:
                    print("Error:please enter valid amount")
                else:
                    buy(input2)
                    print("       ")
                    print("Thank you! Navigating you back to main menu")
                    print("       ")
                break
            elif selected == "2":
                    if input2 == 0:
                        print("Error:please enter valid amount")
                    else:
                        sell(input2)
                        print("       ")
                        print("Thank you! Navigating you back to main menu")
                        print("       ")
                        break
                    
    elif selected == "2":
        done = False
        Blotter = pd.DataFrame(hlist)  
        Blotter = Blotter[["Side", "Ticker","Quantity","Executed_price","Timestamp","Current Balence"]]
        Blotter1 = (Blotter[::-1])
        print(tabulate(Blotter1, headers='keys', tablefmt='psql'))
        print("       ")
        print("Portfolio Allocation")
        print("       ")
        
        pieplot(Blotter)
        
        
                    
    elif selected == "3":
        done = False
        PLdf= pd.DataFrame(pllist) 
        PLdf=PLdf[["Ticker","Quantity","Market","Wap","Upl","Rpl","LR_prediction"]]
        PLdf1 = PLdf[::-1]
        print(tabulate(PLdf1, headers='keys', tablefmt='psql'))
        print("       ")
        print("       ")
        print(" Residual plot for Regression model")
        LR(input1)
        
        print("Prediction model - ARIMA")
        ARIMAmodel(input1)
        #ARMODEL(input1)
        
        print("Predictions for five cosecutive days by ARIMA model")
        ARPredictions(input1)
        
        
        
    elif selected == "4":
        done = True
        print("Thank you for trading with us!!") 
        
        


