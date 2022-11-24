# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:30:39 2022




@author: Araby
"""


#------------------------------------------
# Step 0 - Import Libraies
#------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



#------------------------------------------
# Step 1 - Read Data
#------------------------------------------

bikes = pd.read_csv('hour.csv')


#------------------------------------------
# Step 2 - Prelim analysis and features selection
#------------------------------------------

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['instant','dteday','casual','registered'],axis=1)

bikes_prep.isnull().sum()

#bikes_prep.hist(rwidth=0.9)



#------------------------------------------
# Step 3 - Data Visualisation
#------------------------------------------


plt.subplot(2,2,1)
plt.title("Temp And Demand")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='y')

plt.subplot(2,2,2)
plt.title("ATemp And Demand")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2,c='r')

plt.subplot(2,2,3)
plt.title("Humidity And Demand")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2,c='b')

plt.subplot(2,2,4)
plt.title("Windspeed And Demand")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2,c='g')

plt.tight_layout()

colors=['g','b','y','r']
plt.subplot(3,3,1)
plt.title('Mean by season')
cat_list = bikes_prep['season'].unique()
cat_avg = bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,2)
plt.title('Mean by year')
cat_list = bikes_prep['year'].unique()
cat_avg = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,3)
plt.title('Mean by month')
cat_list = bikes_prep['month'].unique()
cat_avg = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,4)
plt.title('Mean by hour')
cat_list = bikes_prep['hour'].unique()
cat_avg = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,5)
plt.title('Mean by holiday')
cat_list = bikes_prep['holiday'].unique()
cat_avg = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,6)
plt.title('Mean by weekday')
cat_list = bikes_prep['weekday'].unique()
cat_avg = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,7)
plt.title('Mean by workingday')
cat_list = bikes_prep['workingday'].unique()
cat_avg = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)

plt.subplot(3,3,8)
plt.title('Mean by weather')
cat_list = bikes_prep['weather'].unique()
cat_avg = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list,cat_avg,color=colors)
plt.tight_layout()

#------------------------------------------
#  Drop irrelevant features
#------------------------------------------

bikes_prep = bikes_prep.drop(['year','weekday','workingday'],axis=1)



# check for outliers

bikes_prep['demand'].quantile([0.05,.15,.25,.35,.45,.55,.65,.75,.85,.95,.99,1])




#------------------------------------------
# Step 4 - Check Multiple Linear Regression Asspumption
#------------------------------------------

correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()





#------------------------------------------
# Step 5 - Drop irrelevant features
#------------------------------------------

bikes_prep = bikes_prep.drop(['atemp','windspeed'],axis=1)


# Autocorrelation of demand

df1= pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)


#------------------------------------------
# Step 6 - Create / Modify new features
#------------------------------------------


df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist()

plt.figure()
df2.hist()


#bikes_prep['demand'] = np.log(bikes_prep['demand'])

t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns=['t_1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns=['t_2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns=['t_3']


bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)

bikes_prep_lag = bikes_prep_lag.dropna()

#------------------------------------------
# Step 7 - Create Dummy variables and drop frist to avoid dummy varialbles trap   
#------------------------------------------

bikes_prep_lag['season']= bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']= bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']= bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']= bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']= bikes_prep_lag['hour'].astype('category')


bikes_prep_lag = pd.get_dummies(bikes_prep_lag,drop_first=True)


#------------------------------------------
# Step 8 - Create Train and test Split
#------------------------------------------

Y = bikes_prep_lag['demand']
X=bikes_prep_lag.drop(['demand'],axis=1)


tr_size =int( 0.7 * len(X))

x_train = X.values[0:tr_size]
x_test = X.values[tr_size:len(X)]


y_train = Y.values[0:tr_size]
y_test = Y.values[tr_size:len(Y)]




#------------------------------------------
# Step 9 - Fit and Score the model
#------------------------------------------

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(x_train,y_train)

r2_train = std_reg.score(x_train,y_train)
r2_test = std_reg.score(x_test,y_test)



# Create Y predictions

y_predict = std_reg.predict(x_test)


from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_predict))






































































