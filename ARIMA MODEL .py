#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
import itertools
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


# In[2]:


df = pd.read_csv('wind.csv', parse_dates=['datetime'], index_col=['datetime'])
df.head()


# In[3]:


plt.plot(df['powergen'])
plt.show()


# In[4]:


df1 = df.asfreq(freq='D', method='ffill')
df1.head()


# In[5]:


rolmean = df1['powergen'].rolling(window=2).mean()

rolstd = df1['powergen'].rolling(window=2).std()


# In[6]:


plt.figure(figsize=(12,6))
orig = plt.plot(df1['powergen'], color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[7]:


plt.plot(df1['powergen'])
plt.show()


# In[8]:


df1 = df1['powergen'].resample('MS').mean()
df1


# In[9]:


y_train=df1[:len(df1)-11]
y_test= df1[len(df1)-11:]


# In[10]:


y_train[-5:]


# In[11]:


y_train.plot()


# In[12]:


y_test.plot()


# In[13]:


result=adfuller(df['powergen'])
print('ADF test statistics: %f' % result[0])
print('p-value: %f' %result[1])
print('critical values: ')

for key,value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[14]:


fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = sm.graphics.tsa.plot_pacf(y_train, lags=25, ax=ax[0])
ax[1] = sm.graphics.tsa.plot_acf(y_train, lags=25, ax=ax[1])


# In[15]:


ts_decompose = sm.tsa.seasonal_decompose(y_train, model = 'additive')
ts_decompose.plot()
plt.show()


# In[19]:


p = d = q = range(0,3)

pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]

print('SARIMAX: {} x {}'. format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'. format(pdq[2], seasonal_pdq[2]))
print('SARIMAX: {} x {}'. format(pdq[3], seasonal_pdq[3]))
print('SARIMAX: {} x {}'. format(pdq[4], seasonal_pdq[4]))
print('SARIMAX: {} x {}'. format(pdq[5], seasonal_pdq[5]))


# In[21]:


metric_aic_dict= dict()

for pm in pdq:
    for pm_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(y_train,
                                             order=pm,
                                             seasonal_order=pm_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
            model_aic = model.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(pm, pm_seasonal, model_aic.aic))
            metric_aic_dict.update({(pm, pm_seasonal):model_aic.aic})
        except:
            continue
# aic - estimates quality of each model


# In[22]:


{k: v for k, v in sorted(metric_aic_dict.items(), key=lambda x: x[1])}


# In[37]:


model = sm.tsa.statespace.SARIMAX(y_train, order=(1,0,2), seasonal_order=(0,1,1,12),enforce_stationarity=False, enforce_invertibility=False)
model_aic = model.fit()
print(model_aic.summary().tables[1])


# In[39]:


model_aic.plot_diagnostics(figsize=(16,8))
plt.show()


# In[38]:


forecast= model_aic.get_prediction(start=pd.to_datetime('2012-02-01'))
predictions = forecast.predicted_mean

actual = y_test['2012-02-01':]

rmse = np.sqrt((predictions - actual) ** 2).mean()
print('the root mean squared error of our forecasts is {}'.format(round(rmse,2)))


# In[40]:


forecast = model_aic.get_forecast(steps=12)

predictions=forecast.predicted_mean
ci = forecast.conf_int()

fig = df1.plot(label='observed', figsize=(14,7))
fig.set_xlabel('Year')
fig.set_ylabel('power generation')
fig.fill_between(ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1], color='k' ,alpha=.2)

predictions.plot(ax=fig, label='Predictions', alpha=.8, figsize=(14,7))


plt.legend()
plt.show()


# In[41]:


y_train.plot()
predictions.plot()


# In[ ]:




