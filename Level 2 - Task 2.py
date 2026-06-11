# 02 June 2026
# Task 2: Time Series Analysis
# Needed: Python, Pandas, Matplotlib, Statsmodels
# Aim: Plot time series, decompose into trend/seasonality/residuals, apply moving averages

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# 1. LOAD & CLEAN DATA

df = pd.read_csv(r"C:\Users\User\OneDrive\UCT\Year 3\Codveda\Data Set For Task-2026\Data Set For Task\2) Stock Prices Data Set.csv")

df['date'] = pd.to_datetime(df['date']) # converting to datetime format
df = df.dropna() # dropping h rows that have missing values
print(df.shape) # full dataset shape

print(f"Stocks available: {df['symbol'].nunique()} unique symbols")
print(f"Data range: {df['date'].min().date()} to {df['date'].max().date()}")


# 2. ISOLATE ONE STOCK (CHOICE? - AAPL)
# Focusing on Apple(AAPL) as a representative example for the time series analysis

aapl = df[df['symbol'] == 'AAPL'].copy()
aapl = aapl.sort_values('date').set_index('date') # setting the date as our index the time series
aapl_close = aapl['close'] # we shall analyse the closing price

print(f"rows: {len(aapl_close)}")
print(aapl_close.head())


# 3. PLOT THE RAW TIME SERIES

plt.figure(figsize= (14,4))
plt.plot(aapl_close, color = 'darkseagreen', linewidth = 1)
plt.title('AAPL Closing Price (2014 - 2017)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.show()


# 4. MOVING AVERAGE SMOOTHING

aapl['MA30'] = aapl_close.rolling(window = 30).mean()
aapl['MA90'] = aapl_close.rolling(window = 90).mean()

plt.figure(figsize=(14,4))
plt.plot(aapl_close, color = 'darkseagreen', linewidth = 1, alpha = 0.6, label = 'Daily Close')
plt.plot(aapl['MA30'], color =  'gold', linewidth = 1.5, label = '30-Day MA')
plt.plot(aapl['MA90'], color = 'lightsalmon', linewidth = 1.5, label = '90-Day MA')
plt.title('AAPL Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()


# 5. SEASONAL DECOMPOSITION

result = seasonal_decompose(aapl_close, model = 'additive', period = 252)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
 
axes[0].plot(aapl_close, color='steelblue', linewidth=0.8)
axes[0].set_title('Original - AAPL Closing Price')
axes[0].set_ylabel('Price (USD)')
 
axes[1].plot(result.trend, color='seagreen', linewidth=1)
axes[1].set_title('Trend Component')
axes[1].set_ylabel('Price (USD)')
 
axes[2].plot(result.seasonal, color='orange', linewidth=0.8)
axes[2].set_title('Seasonal Component (Yearly Cycle)')
axes[2].set_ylabel('Variation (USD)')
 
axes[3].plot(result.resid, color='mediumvioletred', linewidth=0.6, alpha=0.7)
axes[3].set_title('Residual Component (Random Noise)')
axes[3].set_ylabel('Variation (USD)')
axes[3].set_xlabel('Date')
 
plt.tight_layout(pad=5.0)
plt.show()