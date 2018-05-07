# DATA602-Assignment3

Libraries used :
1) pandas as pd
2) numpy as np
3) json
4) matplotlib.pyplot as plt
5) matplotlib.dates as mdates
6) requests
7) warnings
8) time
9) tabulate
10) urllib.request 
11) BeautifulSoup as soup
12) sklearn.linear_model importing LinearRegression
13) sklearn importing preprocessing ,cross_validation,svm
14) statsmodels.tsa.arima_model importing ARIMA
15) statsmodels.tsa.stattools importing acf, pacf
16) statsmodels.tsa.stattools importing adfuller

program starts with greeting the user and asking to choose from four options:
1)Trade
2)Blotter
3)P/L
4)Quit

As new user : go with option one: Trade:
Type the number 1: and equities will appear :
choose one of the following equities:
Bitcoin
Ethereum
Ripple
Bitcoin cash
Litecoin
           
Type the name of equity you want to trade: For Example:'Bitcoin'

Program will ask for quantity:
enter the amount you want to trade

program will ask you to choose:
1) Buy
2) Sell

Type the number you want to choose:
You will see the the residual plot for LR prediction with confidence as accuracy of prediction.
The predicted value for next dat will apear in P/L table.


2)Blotter:
After trade is done ,user will be navigated back to main menu:
choose option two:
Blotter summary will appear.
it will have folowing columns:
Side   | Ticker   |   Quantity |   Executed_price | Timestamp  |   Current Balence

Also a pie chart showing portfolio allocation will be shown.

3) P/L :
Choosing third option from menu will take you two profit/loss summary

Ticker   |   Quantity |   Market |   Wap |   Upl |   Rpl |   LR_prediction

Prediction column is added for linear regression algorithm.

Second algorithm for ARIMA model will give time series analysis with moving average,ACF,PACF and  will display predictions.

Two Machine algorithms used are:
1) Linear Regression
2) AR/MA/ARIMA model
detail code for ARIMA model and How it works is explained in write up attached.

4) Quit



Scope for improvement:
1) Prediction model svm can be used for better results.
2) program works with flow of one equity at a time(one equity per trade) which can be expanded to more than one.
3) Graphs and visuals can be made more clear and user friendly.
4) Usage if UI interface and CSS Django frame work.


What I learned in comparision with privious assignment:
1) Working on functions
2) Data scarpping with more precision.
3) Loops and Data structure improvement.


Docker :

