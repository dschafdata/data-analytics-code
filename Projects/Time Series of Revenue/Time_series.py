import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose

med=pd.read_csv('C:/Users/dscha/Downloads/D213/medical_time_series.csv',index_col=0, parse_dates=False)
med.shape
med.isnull().any()

plt.xlabel('Days')
plt.ylabel('Revenue in Millions')
plt.plot(med)

def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=30).mean()
    movingSTD=timeseries.rolling(window=30).std()
    orig=plt.plot(timeseries, color='blue', label='Original')
    mean=plt.plot(movingAverage, color='red', label='Rolling Mean')
    std=plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation Shows a Trend')
    plt.show(block=False)
    print ('Results of Dickey-Fuller test: ')
    dftest=adfuller(timeseries['Revenue'], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','No. of Observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s) '%key] = value # Critical Values should always be more than the test statistic
    print(dfoutput)
test_stationarity(med)

med_shift=med-med.shift()
med_shift.dropna(inplace=True)
test_stationarity(med_shift)

plot_acf(med_shift)
plot_pacf(med_shift)

decomp=seasonal_decompose(med, model='additive', period=1)
decomp.plot()

plt.psd(med['Revenue'])

stepwise_fit=auto_arima(med['Revenue'], trace=True, suppress_warnings=True)
stepwise_fit.summary()

print(med.shape)
train=med.iloc[:-30]
test=med.iloc[-30:]
print(train.shape, test.shape)

#AR MODEL (Predicted best from auto-ARIMA)
model = ARIMA(med_shift, order=(1,1,0))
results_ARIMA = model.fit()
plt.plot(med_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.
 fittedvalues-med_shift["Revenue"])**2))
print('Plotting AR model')

#AR MODEL using P=1 from PACF, Q=2 from ACF, and D=0)
model = ARIMA(med_shift, order=(1,0,2))
results_ARIMA = model.fit()
plt.plot(med_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.
 fittedvalues-med_shift["Revenue"])**2))
print('Plotting AR model')

#AR MODEL (Predicted best from auto-ARIMA, with the d value set to 0, since we already shifted the data)
model = ARIMA(med_shift, order=(1,0,0))
results_ARIMA = model.fit()
plt.plot(med_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.
 fittedvalues-med_shift["Revenue"])**2))
print('Plotting AR model')

#AR MODEL using P=1 from PACF, Q=2 from ACF, and D=1)
model = ARIMA(med_shift, order=(1,1,2))
results_ARIMA = model.fit()
plt.plot(med_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.
 fittedvalues-med_shift["Revenue"])**2))
print('Plotting AR model')

#AR MODEL (Best fit based on RSS)
model = ARIMA(train, order=(1,0,2))
results_ARIMA = model.fit()
results_ARIMA.summary()

start=len(train)
end=len(train)+len(test)-1
pred=results_ARIMA.predict(start=start, end=end, typ='levels')
print(pred)
pred.index=med.index[start:end+1]

pred.plot(legend=True)
test['Revenue'].plot(legend=True)

index_future_days = pd.interval_range(start=731, end=821, freq=1, closed='both')
print(index_future_days)

pred=results_ARIMA.predict(start=len(med), end=len(med)+90, typ='levels')
print(pred)

pred.plot(legend=True)

from statsmodels.graphics.tsaplots import plot_predict
plot_predict(results_ARIMA, start=1, end=821)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(train['Revenue'], label='Training')
plt.plot(test['Revenue'], label='Test')
plt.plot(pred, label="Forcasted values for three months")
plt.legend(loc='upper left')
plt.title('ARIMA Predictions')
plt.show()